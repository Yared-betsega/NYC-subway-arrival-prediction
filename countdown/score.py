"""The one scorer for every model (README §3, rule 4). Notebooks never compute the reported numbers themselves.

Input
  data/rows/<split>.parquet            the rows (actual arrival t_b, the MTA's forecast mta_pred, base, ...)
  data/preds/<split>/<MODEL>.parquet   one file per model: trip, p, stop_id_b, pred
                                       pred = predicted arrival as a Unix time (for a residual model: t + base + y_hat)
B0 (the MTA, from mta_pred) and B1 (typical times: t + base) are built in; no file needed.

All models are scored on the same rows: those with an MTA forecast and a prediction from every model. Rows a model
skipped are counted and reported, never silently dropped.

Metrics, per horizon band (minutes ahead at p) and overall:
  MAE (s) with a 95% bootstrap CI that resamples whole trains, share of predictions within 60 s, signed bias
  (+ = predicted later than actual), and each model's MAE minus the MTA's with its own CI (below 0 = beats the MTA).
Splits: division (numbered / lettered / Staten Island), time of day, and without Veterans Day (Nov 11).

Usage:  python score.py val                      → prints the tables, writes results/val_scores.csv
        python score.py val --boot 200           fewer bootstrap draws (faster)
        python score.py test                     ONLY once, after every model is frozen (Nov 22 or later)
"""
import argparse, glob, os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
NY = 'America/New_York'
BANDS = [0, 120, 300, 600, 1200, 1800]
BAND_NAMES = ['0-2', '2-5', '5-10', '10-20', '20-30']
PERIODS = ([0, 6, 10, 16, 20, 24], ['night 0-6', 'AM peak 6-10', 'midday 10-16', 'PM peak 16-20', 'evening 20-24'])
HOLIDAYS = ['2026-11-11']                       # Veterans Day
KEY = ['trip', 'p', 'stop_id_b']


def division(route):
    r = pd.Series(route)
    return np.where(r.isin(['SI', 'SS']), 'Staten Island',
                    np.where(r.str[0].str.isdigit() | (r == 'GS'), 'numbered', 'lettered'))


def load(split, pred_dir=None):
    """Rows of the split plus one prediction column per model, restricted to rows every model covers."""
    rows = pd.read_parquet(os.path.join(DATA, 'rows', f'{split}.parquet'),
                           columns=KEY + ['route', 't', 't_b', 'lead', 'base', 'hour', 'mta_pred'])
    assert not rows.duplicated(KEY).any(), 'duplicate row keys'
    preds = {'B0 MTA': rows.mta_pred.to_numpy(), 'B1 typical': (rows.t + rows.base).to_numpy('float64')}
    for path in sorted(glob.glob(os.path.join(pred_dir or os.path.join(DATA, 'preds', split), '*.parquet'))):
        name = os.path.splitext(os.path.basename(path))[0]
        f = pd.read_parquet(path, columns=KEY + ['pred'])
        assert not f.duplicated(KEY).any(), f'{name}: duplicate keys'
        m = rows[KEY].merge(f, on=KEY, how='left')
        preds[name] = m.pred.to_numpy('float64')
        extra = len(f) - m.pred.notna().sum()
        print(f'{name}: predictions for {m.pred.notna().mean():.2%} of rows'
              + (f' ({extra:,} predictions match no row)' if extra else ''))
    ok = np.ones(len(rows), bool)
    for name, v in preds.items():
        miss = ~np.isfinite(v)
        if miss.any():
            print(f'  {name}: no prediction for {miss.sum():,} rows ({miss.mean():.2%})')
        ok &= ~miss
    print(f'scored on {ok.sum():,} of {len(rows):,} rows ({ok.mean():.2%}) covered by every model\n')
    rows = rows[ok].reset_index(drop=True)
    for name, v in preds.items():
        rows[name] = v[ok] - rows.t_b.to_numpy()                  # signed error, seconds
    rows['band'] = pd.cut(rows.lead, BANDS, right=False, labels=BAND_NAMES)
    rows['day'] = pd.to_datetime(rows.p, unit='s', utc=True).dt.tz_convert(NY).dt.strftime('%Y-%m-%d')
    return rows, list(preds)


def table(rows, models, boot, seed=0):
    """Per band and overall: n, MAE [CI], within 60 s, bias, MAE - MTA [CI]. Bootstrap resamples whole trains."""
    rows = rows[rows.band.notna()]
    tc, trips = pd.factorize(rows.trip)
    groups = [(b, (rows.band == b).to_numpy()) for b in BAND_NAMES] + [('all', np.ones(len(rows), bool))]
    rng = np.random.default_rng(seed)
    draws = [np.bincount(rng.integers(0, len(trips), len(trips)), minlength=len(trips)) for _ in range(boot)]
    W = np.array(draws, 'float64').T if boot else None                # trips x draws
    out = []
    for g, sel in groups:
        n_t = np.bincount(tc[sel], minlength=len(trips)).astype('float64')
        abs_t = {m: np.bincount(tc[sel], np.abs(rows[m].to_numpy()[sel]), minlength=len(trips)) for m in models}
        n_b = n_t @ W if boot else None
        mae_b = {m: (abs_t[m] @ W) / n_b for m in models} if boot else {}
        for m in models:
            e = rows[m].to_numpy()[sel]
            r = {'band': g, 'model': m, 'n': int(sel.sum()), 'MAE': np.abs(e).mean(),
                 'within60': (np.abs(e) <= 60).mean(), 'bias': e.mean()}
            r['MAE - MTA'] = r['MAE'] - np.abs(rows[models[0]].to_numpy()[sel]).mean()
            if boot:
                r['MAE lo'], r['MAE hi'] = np.percentile(mae_b[m], [2.5, 97.5])
                r['diff lo'], r['diff hi'] = np.percentile(mae_b[m] - mae_b[models[0]], [2.5, 97.5])
            out.append(r)
    return pd.DataFrame(out)


def show(t, boot):
    def ci(v, lo, hi):
        return f'{v:6.1f} [{lo:5.1f}, {hi:5.1f}]' if boot else f'{v:6.1f}'
    disp = pd.DataFrame({
        'band': t.band, 'model': t.model, 'n': t.n.map('{:,}'.format),
        'MAE s [95% CI]': [ci(*x) for x in zip(t.MAE, t.get('MAE lo', t.MAE), t.get('MAE hi', t.MAE))],
        'within 60 s': (100 * t.within60).map('{:.1f}%'.format), 'bias s': t.bias.map('{:+.1f}'.format),
        'MAE - MTA [95% CI]': [ci(*x) for x in zip(t['MAE - MTA'], t.get('diff lo', t.MAE), t.get('diff hi', t.MAE))]})
    print(disp.to_string(index=False))


def split_table(rows, models, col):
    """MAE by band for each value of a split column (no CIs; for spotting where a model wins or loses)."""
    s = rows[rows.band.notna()].melt(id_vars=[col, 'band'], value_vars=models, var_name='model', value_name='e')
    s['e'] = s.e.abs()
    return s.pivot_table(index=[col, 'model'], columns='band', values='e', aggfunc='mean', observed=True).round(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('split', choices=['train', 'val', 'test'])
    ap.add_argument('--preds', help='folder of prediction files (default data/preds/<split>)')
    ap.add_argument('--boot', type=int, default=1000, help='bootstrap draws (0 = no CIs)')
    args = ap.parse_args()

    rows, models = load(args.split, args.preds)
    print(f'{args.split}: {rows.day.min()} to {rows.day.max()}, {rows.trip.nunique():,} trains\n')
    t = table(rows, models, args.boot)
    show(t, args.boot)

    rows['division'] = division(rows.route)
    rows['period'] = pd.cut(rows.hour, PERIODS[0], right=False, labels=PERIODS[1])
    for col in ['division', 'period']:
        print(f'\nMAE (s) by {col}:')
        print(split_table(rows, models, col).to_string())
    hol = rows.day.isin(HOLIDAYS)
    if hol.any():
        print(f'\nwithout {", ".join(HOLIDAYS)} ({hol.sum():,} rows removed):')
        show(table(rows[~hol], models, 0), 0)

    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    out = os.path.join(HERE, 'results', f'{args.split}_scores.csv')
    t.round(3).to_csv(out, index=False)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
