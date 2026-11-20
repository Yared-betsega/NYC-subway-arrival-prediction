"""True arrival times from the recordings: arrival = the train's first STOPPED_AT report at a stop.

Reads vehicles.csv.gz from both recorders (data/vm = main, data/raw = PC backup) and pools their reports, so a
gap in one copy is filled by the other and the two 30-s poll grids (about 10 s apart) give finer timing.
Per (trip_id, start_date, stop_id) it keeps:
  arrived   train's own timestamp on its first STOPPED_AT report at the stop (the label)
  left      timestamp of its last STOPPED_AT report there (end of dwell, approximately)
  known     first poll that showed it, i.e. when we could have known it (used by rows.py for the leakage check)
  src       which copy saw it: vm, pc or both
Damaged files (see repair.py) are read through the salvage path into a temp copy; the originals are not touched.
Trips crossing midnight appear in two day files; the earliest report wins.

Usage:  python labels.py [--start 2026-10-31] [--end 2026-11-14]      → data/labels/arrivals.parquet
"""
import argparse, os, tempfile
from datetime import date, timedelta

import pandas as pd

from repair import is_damaged, salvage

HERE = os.path.dirname(os.path.abspath(__file__))
COPIES = {'vm': os.path.join(HERE, 'data', 'vm'), 'pc': os.path.join(HERE, 'data', 'raw')}
COLS = ['poll', 'trip_id', 'start_date', 'route_id', 'stop_id', 'status', 'veh_ts']


def read_vehicles(path, tmpdir):
    if is_damaged(path):
        fixed = os.path.join(tmpdir, 'vehicles.csv.gz')
        kept, faults = salvage(path, fixed)
        print(f'   {path}: damaged, salvaged {kept:,} rows ({faults} broken stretch(es) skipped)')
        path = fixed
    v = pd.read_csv(path, usecols=COLS, dtype={'trip_id': str, 'start_date': str, 'route_id': str, 'stop_id': str})
    return v[(v.status == 1) & (v.veh_ts > 0)].drop(columns='status')


def day_reports(day, tmpdir):
    parts = []
    for name, root in COPIES.items():
        path = os.path.join(root, str(day), 'vehicles.csv.gz')
        if os.path.exists(path):
            parts.append(read_vehicles(path, tmpdir).assign(src=name))
    return pd.concat(parts, ignore_index=True) if parts else None


def summarize(reports):
    key = ['trip_id', 'start_date', 'stop_id']
    g = reports.groupby(key, sort=False)
    out = g.agg(route_id=('route_id', 'first'), arrived=('veh_ts', 'min'), left=('veh_ts', 'max'), known=('poll', 'min'))
    srcs = g.src.agg(lambda s: 'both' if s.nunique() > 1 else s.iloc[0])
    return out.assign(src=srcs).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-10-31')
    ap.add_argument('--end', default='2026-11-14')
    ap.add_argument('--out', default=os.path.join(HERE, 'data', 'labels', 'arrivals.parquet'))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    days, day = [], date.fromisoformat(args.start)
    while day <= date.fromisoformat(args.end):
        with tempfile.TemporaryDirectory(dir=os.path.join(HERE, 'data')) as tmp:
            r = day_reports(day, tmp)
        if r is None:
            print(f'{day}: no recordings')
        else:
            s = summarize(r)
            print(f'{day}: {len(s):,} arrivals  (vm {(s.src == "vm").mean():.1%}, pc {(s.src == "pc").mean():.1%}, '
                  f'both {(s.src == "both").mean():.1%})')
            days.append(s)
        day += timedelta(days=1)

    a = pd.concat(days, ignore_index=True)
    a = a.sort_values('arrived').groupby(['trip_id', 'start_date', 'stop_id'], sort=False).agg(
        route_id=('route_id', 'first'), arrived=('arrived', 'min'), left=('left', 'max'), known=('known', 'min'),
        src=('src', 'first')).reset_index()
    a['direction'] = a.stop_id.str[-1]
    a.to_parquet(args.out, index=False)
    print(f'wrote {len(a):,} arrivals to {args.out}')


if __name__ == '__main__':
    main()
