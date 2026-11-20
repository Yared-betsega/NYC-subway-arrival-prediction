"""Download subwaydata.nyc's daily archive of actual stop times (the pretraining history).

Each day is one file, kept compressed: data/history/subwaydatanyc_YYYY-MM-DD_csv.tar.xz (~1 MB). Inside are
  subwaydatanyc_YYYY-MM-DD_stop_times.csv  trip_uid, stop_id, track, arrival_time, departure_time, last_observed, marked_past
  subwaydatanyc_YYYY-MM-DD_trips.csv       trip_uid, trip_id, route_id, direction_id, start_time, vehicle_id, ...
Arrival times use the "last reported" definition (30-75 s later than STOPPED_AT; correction.py fixes that).
Files already downloaded are skipped, so it is safe to rerun.

Usage:  python get_history.py [--start 2026-05-01] [--end 2026-11-13] [--out data/history]
"""
import argparse, os, time
from datetime import date, timedelta

import requests

URL = 'https://subwaydata.nyc/data/subwaydatanyc_{}_csv.tar.xz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-05-01')
    ap.add_argument('--end', default='2026-11-13')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'history'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    day, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    session, got, skipped, missing = requests.Session(), 0, 0, []
    while day <= end:
        path = os.path.join(args.out, f'subwaydatanyc_{day}_csv.tar.xz')
        if os.path.exists(path):
            skipped += 1
        else:
            for attempt in range(3):
                try:
                    r = session.get(URL.format(day), timeout=60)
                    if r.status_code == 404:
                        missing.append(str(day))
                        break
                    r.raise_for_status()
                    with open(path + '.part', 'wb') as f:
                        f.write(r.content)
                    os.replace(path + '.part', path)
                    got += 1
                    break
                except requests.RequestException as e:
                    if attempt == 2:
                        missing.append(f'{day} ({e})')
                    time.sleep(5)
            time.sleep(0.5)                                   # be polite to a free, volunteer-run site
        day += timedelta(days=1)
    print(f'downloaded {got}, already had {skipped}, missing {len(missing)}')
    for m in missing:
        print('  missing', m)


if __name__ == '__main__':
    main()
