"""Per-station offset that moves history arrival times onto our STOPPED_AT definition.

History ("last reported") times run late: about 30 s on numbered lines, and 57-98 s on lettered lines depending on the
station. For each stop_id (station + direction) this takes the median of (history time - STOPPED_AT arrival) over the
same trains, fitted on the TRAINING WEEK ONLY. Stops with fewer than MIN_N matches use their line's median instead.
rows.py subtracts the offset from every history time before building rows.

It then checks the fit on the validation week: after correction the leftover offset should be close to 0 s.

Usage:  python correction.py      → data/labels/correction.parquet  (stop_id, route_id, offset, n, level)
"""
import os

import numpy as np
import pandas as pd

from get_history import load_day

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, 'data', 'labels')
TRAIN = ('2026-10-31', '2026-11-07')
VAL = ('2026-11-08', '2026-11-13')            # history ends Nov 13
MIN_N = 30
NY = 'America/New_York'


def ny_date(t):
    return pd.to_datetime(t, unit='s', utc=True).dt.tz_convert(NY).dt.strftime('%Y-%m-%d')


def matched(labels, first, last):
    """History and STOPPED_AT times for the same train at the same stop, on days first..last.
    Origin stops are left out (history only has a departure there, not an arrival)."""
    days = pd.date_range(first, last).strftime('%Y-%m-%d')
    h = pd.concat([load_day(d) for d in days], ignore_index=True)
    h = h[h.arrival_time.notna()].copy()
    h['day'] = ny_date(h.t)
    lab = labels[labels.day.isin(days)]
    m = lab.merge(h[['trip_id', 'stop_id', 'day', 't']], on=['trip_id', 'stop_id', 'day'])
    m['diff'] = m.t - m.arrived
    return m[m['diff'].abs() < 600]


def division(route):
    return np.where(route.str[0].str.isdigit() | (route == 'GS'), 'numbered', 'lettered')


def main():
    labels = pd.read_parquet(os.path.join(LABELS, 'arrivals.parquet'),
                             columns=['trip_id', 'stop_id', 'route_id', 'arrived'])
    labels['day'] = ny_date(labels.arrived)

    tr = matched(labels, *TRAIN)
    by_stop = tr.groupby('stop_id').agg(route_id=('route_id', lambda s: s.mode().iloc[0]),
                                        offset=('diff', 'median'), n=('diff', 'size'))
    by_route = tr.groupby('route_id')['diff'].median()
    corr = by_stop.reset_index()
    corr['level'] = np.where(corr.n >= MIN_N, 'station', 'line')
    corr.loc[corr.level == 'line', 'offset'] = corr.route_id.map(by_route)
    corr['offset'] = corr.offset.round().astype(int)
    out = os.path.join(LABELS, 'correction.parquet')
    corr.to_parquet(out, index=False)
    by_route.round().astype(int).rename('offset').reset_index().to_parquet(
        os.path.join(LABELS, 'correction_by_line.parquet'), index=False)
    print(f'training week: {len(tr):,} matched arrivals; {len(corr)} stops '
          f'({(corr.level == "station").sum()} with their own offset, {(corr.level == "line").sum()} using the line median)')
    print(corr.assign(div=division(corr.route_id)).groupby('div').offset.describe(percentiles=[.1, .5, .9])
          [['count', '10%', '50%', '90%']].round(0).to_string())

    # check on the validation week: raw vs corrected leftover offset
    va = matched(labels, *VAL)
    off = va.stop_id.map(corr.set_index('stop_id').offset).fillna(va.route_id.map(by_route))
    va = va.assign(raw=va['diff'], left=va['diff'] - off, div=division(va.route_id)).dropna(subset=['left'])
    summary = va.groupby('div').agg(n=('raw', 'size'), raw_median=('raw', 'median'), left_median=('left', 'median'),
                                    left_mean_abs=('left', lambda s: s.abs().mean()))
    print(f'\nvalidation week ({VAL[0]} – {VAL[1]}), seconds:')
    print(summary.round(1).to_string())
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
