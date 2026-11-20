"""Build the feature rows the models train on (README §2).

One row = one train x one later target stop x one prediction moment p. At p, the train's anchor is the furthest stop it
was confirmed at by p; the target is a stop up to 30 min ahead. Every feature uses only events known at or before p,
and this is asserted for every row (leakage check).

Sources, all through the same build() so the files cannot drift apart:
  history.parquet  May 1 - Oct 30, subwaydata.nyc times shifted onto STOPPED_AT time by correction.py. History has no
                   "known at" time, so each arrival is treated as known after a confirmation delay drawn from the
                   recordings. A sample of trips, one moment every 15 min.
  train.parquet    p on Oct 31 - Nov 7, arrivals from labels.py, plus the MTA's forecast at p (mta_pred, scoring only).
  val.parquet      p on Nov 8 - 14, same.
Typical segment times ("base") are medians over Sep 1 - Oct 30 of history, by line, stop pair, time band and weekend.
The test week is never read. Everything is built and written one day at a time, so memory stays around 2 GB.

Usage:  python rows.py                 full build → data/rows/{history,train,val}.parquet
        python rows.py --days 2        quick test on 2 days per part → data/rows_test/
"""
import argparse, gzip, os, resource, time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from get_history import load_day

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
NY = 'America/New_York'
MEDIAN_DAYS = ('2026-09-01', '2026-10-30')
HISTORY_DAYS = ('2026-05-01', '2026-10-30')
LIVE_DAYS = ('2026-10-31', '2026-11-14')
VAL_FROM = '2026-11-08'
HORIZON = 1800                 # predict up to 30 min ahead
J = 30                         # look at most this many stops ahead of the anchor
K = 8                          # past segment delays kept for the LSTM (sd1..sd8)
HIST_EVERY, HIST_FRAC = 900, 0.15
LIVE_EVERY = 600
POLL_COPY = {'2026-10-31': 'raw'}          # which recording supplies the MTA forecast; default vm (PC fuller on Oct 31)
BANDS = [0, 120, 300, 600, 1200, 1800]
FEATURES = ['base', 'n_stops', 'elapsed', 'overdue', 'hour', 'weekend', 'dev1', 'dev3', 'gap',
            'ahead_dev', 'ahead_pace', 'ahead_frac', 'ahead_since'] + [f'sd{k}' for k in range(1, K + 1)]


def days(first, last):
    return list(pd.date_range(first, last).strftime('%Y-%m-%d'))


def bands(t):
    """Hour of day, time band (0-4) and weekend flag of Unix times, in New York time."""
    loc = pd.to_datetime(np.asarray(t), unit='s', utc=True).tz_convert(NY)
    hour = np.asarray(loc.hour + loc.minute / 60, 'float32')
    return hour, np.searchsorted([6, 10, 16, 20], np.asarray(loc.hour), side='right'), np.asarray(loc.day_of_week >= 5, 'int64')


class Vocab(dict):
    """Stable integer codes for stop and route ids."""
    def codes(self, s):
        for v in pd.unique(s):
            if v not in self:
                self[v] = len(self)
        return s.map(self).to_numpy('int64')


def key1(r, a, b, hb, wk):
    return ((r * 4096 + a) * 4096 + b) * 16 + hb * 2 + wk


class Medians:
    """Typical time between two stops: by (line, stop pair, time band, weekend) if seen at least 5 times, else by stop pair."""
    def __init__(self, k1, seg):
        s = pd.Series(seg).groupby(k1).agg(['median', 'size'])
        self.m1 = s['median'][s['size'] >= 5]
        self.m2 = pd.Series(seg).groupby((k1 // 16) % (4096 * 4096)).median()      # same key without line and time

    def lookup(self, r, a, b, hb, wk):
        v = self.m1.reindex(key1(r, a, b, hb, wk)).to_numpy()
        v = np.where(np.isnan(v), self.m2.reindex(a * 4096 + b).to_numpy(), v)
        return np.where(b < 0, np.nan, v)


def history_events(day, corr):
    """One history day: arrivals only (origin stops have just a departure), shifted onto STOPPED_AT time."""
    h = load_day(day)
    if h is None:
        return None
    h = h[h.arrival_time.notna()]
    off = h.stop_id.map(corr['stop']).fillna(h.route_id.map(corr['line'])).fillna(0)
    return pd.DataFrame({'trip': h.trip_uid.to_numpy(), 'route_id': h.route_id.to_numpy(),
                         'stop_id': h.stop_id.to_numpy(), 't': (h.t - off).astype('int64').to_numpy()})


def segment_medians(corr, stops, routes, limit):
    """Stop-to-stop times from history, for neighbours 1-3 stops apart (recordings can miss a short stop)."""
    k1s, segs = [], []
    for d in days(*MEDIAN_DAYS)[-limit:] if limit else days(*MEDIAN_DAYS):
        ev = history_events(d, corr)
        if ev is None:
            continue
        ev = ev.sort_values(['trip', 't'], kind='stable')
        tr, t = ev.trip.to_numpy(), ev.t.to_numpy()
        a, r = stops.codes(ev.stop_id), routes.codes(ev.route_id)
        _, hb, wk = bands(t)
        for k in (1, 2, 3):
            ok = np.zeros(len(t), bool)
            ok[:-k] = tr[k:] == tr[:-k]
            seg = np.zeros(len(t), 'int64')
            seg[:-k] = t[k:] - t[:-k]
            ok &= (seg > 0) & (seg < HORIZON)
            b = np.zeros_like(a)
            b[:-k] = a[k:]
            k1s.append(key1(r, a, b, hb, wk)[ok]); segs.append(seg[ok].astype('int16'))
    return Medians(np.concatenate(k1s), np.concatenate(segs))


def build(ev, every, frac, rng, med, stops, routes, polls=None):
    """Rows for one day of events (trip, route_id, stop_id, t, known). Moments every `every` s per trip (random phase),
    for a `frac` sample of trips; with `polls`, moments snap to the next recorded poll (so the MTA forecast exists)."""
    ev = ev.sort_values(['trip', 't'], kind='stable').reset_index(drop=True)
    n = len(ev)
    tc, trip_names = pd.factorize(ev.trip)
    t = ev.t.to_numpy('int64')
    first = np.r_[True, tc[1:] != tc[:-1]]
    last = np.r_[first[1:], True]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    seq = np.arange(n) - start
    known = pd.Series(ev.known.to_numpy('int64')).groupby(tc).cummax().to_numpy()   # a stop counts as known only once all earlier ones are
    a, r = stops.codes(ev.stop_id), routes.codes(ev.route_id)
    assert len(stops) < 4096 and len(routes) < 64
    _, hb, wk = bands(t)

    nxt = np.where(last, -1, np.r_[a[1:], -1])
    seg = np.where(last, np.nan, np.r_[t[1:], 0] - t)
    segmed = med.lookup(r, a, nxt, hb, wk)                        # typical time to the train's next recorded stop
    miss = (~last & np.isnan(segmed)).astype('int64')

    def before(x):                                                # sum over the trip's earlier segments
        cs = np.cumsum(x)
        return cs - x - (cs[start] - x[start])

    def shift(x, k):                                              # value k stops back on the same trip
        return np.where(seq >= k, np.r_[np.full(k, np.nan), x[:-k]], np.nan)

    cum_med, cum_miss = before(np.nan_to_num(segmed)), before(miss)
    sd = seg - segmed
    dev = {k: (t - shift(t, k)) - (cum_med - shift(cum_med, k)) for k in (1, 3)}

    order = np.lexsort((t, a, r))                                 # previous train of the same line at the same stop
    prev = np.full(n, -1)
    same = (r[order][1:] == r[order][:-1]) & (a[order][1:] == a[order][:-1])
    prev[order[1:][same]] = order[:-1][same]

    # prediction moments
    ntr = len(trip_names)
    t0, t1 = known[first], t[last]
    phase = rng.uniform(0, every, ntr)
    cnt = np.where(rng.random(ntr) < frac, np.floor((t1 - t0 - phase) / every) + 1, 0).clip(0).astype('int64')
    mt = np.repeat(np.arange(ntr), cnt)
    within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    p = (t0[mt] + phase[mt] + every * within).astype('int64')
    if polls is not None:
        i = np.minimum(np.searchsorted(polls, p), len(polls) - 1)
        ok = (polls[i] >= p) & (polls[i] - p <= 45)
        mt, p = mt[ok], polls[i][ok]

    right = pd.DataFrame({'tc': tc, 'k': known, 'i': np.arange(n)}).sort_values('k')

    def latest(trips, at):                                        # furthest event of each trip known by `at`
        left = pd.DataFrame({'tc': trips, 'p': at, 'm': np.arange(len(at))}).sort_values('p')
        x = pd.merge_asof(left, right, left_on='p', right_on='k', by='tc', direction='backward').sort_values('m')
        return x.i.fillna(-1).to_numpy('int64')

    ra = latest(mt, p)
    mt, p, ra = mt[ra >= 0], p[ra >= 0], ra[ra >= 0]
    pi = prev[ra]
    pi = np.where((pi >= 0) & (known[np.maximum(pi, 0)] <= p), pi, -1)
    pc = latest(np.where(pi >= 0, tc[np.maximum(pi, 0)], -1), p)

    # targets: later stops on the same trip, not yet reached at p, within 30 min, no unknown segment in between
    mi, rb = [], []
    for j in range(1, J + 1):
        b = np.minimum(ra + j, n - 1)
        ok = (ra + j < n) & (tc[b] == mt) & (t[b] >= p) & (t[b] - p < HORIZON) & (cum_miss[b] == cum_miss[ra])
        mi.append(np.nonzero(ok)[0]); rb.append(b[ok])
    mi, rb = np.concatenate(mi), np.concatenate(rb)
    A, P, PI, PC = ra[mi], p[mi], pi[mi], pc[mi]
    base = cum_med[rb] - cum_med[A]

    # train ahead: did it reach the target by p (its time over the same stretch is known), else how far along it is
    keys, firstpos = np.unique(tc * 4096 + a, return_index=True)
    want = np.where(PI >= 0, tc[np.maximum(PI, 0)] * 4096 + a[rb], -1)
    pos = np.minimum(np.searchsorted(keys, want), len(keys) - 1)
    pb = np.where(keys[pos] == want, firstpos[pos], -1)
    reached = (pb >= 0) & (known[np.maximum(pb, 0)] <= P)
    PIc, PBc, PCc = np.maximum(PI, 0), np.maximum(pb, 0), np.maximum(PC, 0)
    on_way = ~reached & (PI >= 0) & (PC >= 0)
    stretch = cum_med[PCc] - cum_med[PIc]

    assert (known[A] <= P).all(), 'anchor not known at p'
    assert (known[PIc][PI >= 0] <= P[PI >= 0]).all(), 'train ahead not known at p'
    assert (known[PBc][reached] <= P[reached]).all() and (known[PCc][on_way] <= P[on_way]).all(), 'train ahead leaks'

    hour, _, weekend = bands(P)
    out = pd.DataFrame({
        'trip': trip_names[tc[A]], 'p': P, 'stop_id_b': ev.stop_id.to_numpy()[rb],
        'route': ev.route_id.to_numpy()[A], 'direction': (ev.stop_id.str[-1].to_numpy()[A] == 'S').astype('int8'),
        't': t[A], 't_b': t[rb], 'lead': t[rb] - P, 'y': (t[rb] - t[A]) - base,
        'base': base, 'n_stops': seq[rb] - seq[A], 'elapsed': P - t[A], 'overdue': P - t[A] - segmed[A],
        'hour': hour, 'weekend': weekend, 'dev1': dev[1][A], 'dev3': dev[3][A],
        'gap': np.where(PI >= 0, t[A] - t[PIc], np.nan),
        'ahead_dev': np.where(reached, (t[PBc] - t[PIc]) - (cum_med[PBc] - cum_med[PIc]), np.nan),
        'ahead_pace': np.where(on_way, (t[PCc] - t[PIc]) - stretch, np.nan),
        'ahead_frac': np.where(on_way, stretch / np.maximum(base, 1), np.nan),
        'ahead_since': np.where(on_way, P - t[PCc], np.nan),
        **{f'sd{k}': shift(sd, k)[A] for k in range(1, K + 1)}})
    return out


def finalize(df):
    df[FEATURES + ['y', 'lead']] = df[FEATURES + ['y', 'lead']].astype('float32')
    df['n_stops'] = df.n_stops.astype('int16')
    df['weekend'] = df.weekend.astype('int8')
    return df


def attach_mta(rows, copy_of_day):
    """mta_pred = the MTA's predicted arrival at stop_id_b, from the poll at p (scoring only, never an input)."""
    rows['mta_pred'] = np.nan
    trip_id, start_date = rows.trip.str.split('|', n=1).str[0], rows.trip.str.split('|', n=1).str[1]
    day = pd.to_datetime(rows.p, unit='s', utc=True).dt.tz_convert(NY).dt.strftime('%Y-%m-%d')
    for d, idx in rows.groupby(day).groups.items():
        path = os.path.join(DATA, copy_of_day(d), d, 'predictions.csv.gz')
        need = rows.loc[idx, ['p']].assign(trip_id=trip_id[idx], start_date=start_date[idx], stop_id=rows.stop_id_b[idx])
        wanted = pa.array((need.p.astype(str) + '|' + need.trip_id + '|' + need.start_date + '|' + need.stop_id).unique())
        types = {'poll': pa.int64(), 'trip_id': pa.string(), 'start_date': pa.string(), 'stop_id': pa.string(),
                 'arr': pa.float64(), 'dep': pa.float64()}
        found = []
        with gzip.open(path, 'rb') as fh:                         # streamed; only the exact (poll, train, stop) rows needed are kept
            for batch in pacsv.open_csv(fh, read_options=pacsv.ReadOptions(block_size=8 << 20, use_threads=False),
                                        convert_options=pacsv.ConvertOptions(include_columns=list(types), column_types=types)):
                key = pc.binary_join_element_wise(pc.cast(batch.column('poll'), pa.string()), batch.column('trip_id'),
                                                  batch.column('start_date'), batch.column('stop_id'), '|')
                found.append(batch.filter(pc.is_in(key, value_set=wanted)).to_pandas())
        f = pd.concat(found)
        f['mta_pred'] = f.arr.where(f.arr > 0, f.dep)
        f = f[f.mta_pred > 0]                                     # the feed sometimes sends 0 for both: no forecast.drop_duplicates(['poll', 'trip_id', 'start_date', 'stop_id'])
        m = need.reset_index().merge(f, left_on=['p', 'trip_id', 'start_date', 'stop_id'],
                                     right_on=['poll', 'trip_id', 'start_date', 'stop_id'], how='left')
        rows.loc[m['index'], 'mta_pred'] = m.mta_pred.to_numpy()
        print(f'   {d}: MTA forecast found for {m.mta_pred.notna().mean():.1%} of {len(m):,} rows')
    return rows


def mem():
    return f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB peak'


class Writer:
    """Appends tables to one parquet file."""
    def __init__(self, path):
        self.path, self.w, self.n = path, None, 0

    def write(self, df):
        table = pa.Table.from_pandas(df, preserve_index=False)
        self.w = self.w or pq.ParquetWriter(self.path, table.schema)
        self.w.write_table(table.cast(self.w.schema))
        self.n += len(df)

    def close(self):
        if self.w:
            self.w.close()


def summary(name, rows):
    band = pd.cut(rows.lead, BANDS, right=False, labels=['0-2', '2-5', '5-10', '10-20', '20-30'])
    s = rows.assign(band=band, B1=rows.y.abs())
    cols = {'rows': ('y', 'size'), 'B1 (medians) MAE': ('B1', 'mean')}
    if 'mta_pred' in rows:
        s = s[s.mta_pred.notna()].assign(B0=(s.mta_pred - s.t_b).abs())
        cols['B0 (MTA) MAE'] = ('B0', 'mean')
    print(f'{name}: {len(rows):,} rows, {rows.trip.nunique():,} trips; error by minutes ahead (s):')
    print(s.groupby('band', observed=True).agg(**cols).round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=0, help='quick test: only this many days per part')
    args = ap.parse_args()
    out_dir = os.path.join(DATA, 'rows_test' if args.days else 'rows')
    os.makedirs(out_dir, exist_ok=True)
    rng, t_start = np.random.default_rng(0), time.time()

    c = pd.read_parquet(os.path.join(DATA, 'labels', 'correction.parquet'))
    cl = pd.read_parquet(os.path.join(DATA, 'labels', 'correction_by_line.parquet'))
    corr = {'stop': c.set_index('stop_id').offset, 'line': cl.set_index('route_id').offset}
    stops, routes = Vocab(), Vocab()
    med = segment_medians(corr, stops, routes, args.days)
    print(f'medians: {len(med.m1):,} detailed, {len(med.m2):,} stop pairs ({time.time() - t_start:.0f} s, {mem()})')

    lab = pd.read_parquet(os.path.join(DATA, 'labels', 'arrivals.parquet'),
                          columns=['trip_id', 'start_date', 'route_id', 'stop_id', 'arrived', 'known'])
    lag = (lab.known - lab.arrived).clip(0, 120).to_numpy()

    # history, one day at a time
    w = Writer(os.path.join(out_dir, 'history.parquet'))
    for d in days(*HISTORY_DAYS)[:args.days] if args.days else days(*HISTORY_DAYS):
        ev = history_events(d, corr)
        if ev is None:
            continue
        ev['known'] = ev.t + rng.choice(lag, len(ev))
        rows = finalize(build(ev, HIST_EVERY, HIST_FRAC, rng, med, stops, routes))
        w.write(rows)
    w.close()
    print(f'history: {w.n:,} rows ({time.time() - t_start:.0f} s, {mem()})')
    summary('history (last day)', rows)

    # recordings, one service day at a time. Rows go to train/val by the date of p; a day's rows are written once no
    # later service day can add to that date (trips after midnight belong to the previous service day).
    copy_of_day = lambda d: POLL_COPY.get(d, 'vm')
    live_days = days(*LIVE_DAYS)
    if args.days:
        live_days = live_days[:args.days] + [VAL_FROM]
    polls = np.unique(np.concatenate([pd.read_csv(os.path.join(DATA, copy_of_day(d), d, 'vehicles.csv.gz'),
                                                  usecols=['poll']).poll.unique() for d in live_days]))
    lab['trip'] = lab.trip_id + '|' + lab.start_date
    lab['service_day'] = pd.to_datetime(lab.start_date).dt.strftime('%Y-%m-%d')
    lab = lab[lab.service_day.isin(set(live_days) | {(pd.Timestamp(live_days[0]) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')})]
    first_stop = lab.groupby('trip').arrived.transform('min') == lab.arrived
    origins = set(load_day(HISTORY_DAYS[1]).query('arrival_time != arrival_time').stop_id)
    lab = lab[~(first_stop & lab.stop_id.isin(origins))]          # same as history: origins never anchor or target
    writers = {'train': Writer(os.path.join(out_dir, 'train.parquet')), 'val': Writer(os.path.join(out_dir, 'val.parquet'))}
    pending, shown = [], {}
    for sd, ev in lab.groupby('service_day'):
        ev = ev.rename(columns={'arrived': 't'})[['trip', 'route_id', 'stop_id', 't', 'known']]
        rows = finalize(build(ev, LIVE_EVERY, 1.0, rng, med, stops, routes, polls))
        print(f'service day {sd}: {len(rows):,} rows ({time.time() - t_start:.0f} s, {mem()})')
        rows['pday'] = pd.to_datetime(rows.p, unit='s', utc=True).dt.tz_convert(NY).dt.strftime('%Y-%m-%d')
        pending.append(rows[rows.pday.isin(live_days)])
        allp = pd.concat(pending, ignore_index=True)
        ready = allp.pday <= sd
        for d, part in allp[ready].groupby('pday'):
            part = attach_mta(part.drop(columns='pday').reset_index(drop=True), copy_of_day)
            name = 'val' if d >= VAL_FROM else 'train'
            writers[name].write(part)
            shown[name] = part
        pending = [allp[~ready]]
    for name, wr in writers.items():
        wr.close()
        print(f'{name}: {wr.n:,} rows')
        summary(f'{name} (last day)', shown[name])
    print(f'done in {time.time() - t_start:.0f} s, {mem()} → {out_dir}')


if __name__ == '__main__':
    main()
