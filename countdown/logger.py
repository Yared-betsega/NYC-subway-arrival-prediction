"""Record the MTA's live subway predictions and train positions every 30 seconds.

For each poll of the 8 GTFS-Realtime feeds it appends to that day's files:
  data/raw/YYYY-MM-DD/predictions.csv.gz  one row per (train, upcoming stop): the MTA's predicted arrival/departure
  data/raw/YYYY-MM-DD/vehicles.csv.gz     one row per train: the stop it is at or heading to, and its status
                                          (STOPPED_AT = standing at the stop; this is our arrival label)
Days follow New York time. A restart appends to the current day's files, so a crash only leaves a gap.

Usage:  python logger.py [--out data/raw] [--interval 30]
"""
import argparse, csv, gzip, logging, os, signal, time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from google.transit import gtfs_realtime_pb2

BASE = 'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2F'
FEEDS = ['gtfs', 'gtfs-ace', 'gtfs-bdfm', 'gtfs-g', 'gtfs-jz', 'gtfs-nqrw', 'gtfs-l', 'gtfs-si']
NY = ZoneInfo('America/New_York')
PRED_COLS = ['poll', 'feed', 'feed_ts', 'trip_id', 'start_date', 'route_id', 'idx', 'stop_id', 'arr', 'dep']
VEH_COLS = ['poll', 'feed', 'feed_ts', 'trip_id', 'start_date', 'route_id', 'stop_id', 'status', 'seq', 'veh_ts']


class DayFiles:
    """Gzip CSV writers for the current New York day; rolls over at midnight."""

    def __init__(self, root):
        self.root, self.day, self.files = root, None, {}

    def writers(self, now):
        day = datetime.fromtimestamp(now, NY).strftime('%Y-%m-%d')
        if day != self.day:
            self.close()
            folder = os.path.join(self.root, day)
            os.makedirs(folder, exist_ok=True)
            for name, cols in (('predictions', PRED_COLS), ('vehicles', VEH_COLS)):
                path = os.path.join(folder, f'{name}.csv.gz')
                new = not os.path.exists(path)
                fh = gzip.open(path, 'at', newline='')   # each restart adds a gzip member; readers handle that
                w = csv.writer(fh)
                if new:
                    w.writerow(cols)
                self.files[name] = (fh, w)
            self.day = day
            logging.info('writing to %s', folder)
        return self.files['predictions'][1], self.files['vehicles'][1]

    def flush(self):
        for fh, _ in self.files.values():
            fh.flush()

    def close(self):
        for fh, _ in self.files.values():
            fh.close()
        self.files = {}


def fetch(session, feed):
    """Download and parse one feed; None if it fails (the poll records what it could get)."""
    for attempt in range(2):
        try:
            r = session.get(BASE + feed, timeout=10)
            r.raise_for_status()
            msg = gtfs_realtime_pb2.FeedMessage()
            msg.ParseFromString(r.content)
            return msg
        except Exception as e:                    # network error, HTTP error, truncated protobuf
            if attempt == 1:
                logging.warning('%s failed: %s', feed, e)
    return None


def poll_once(session, files):
    now = int(time.time())
    preds, vehs = files.writers(now)
    counts = {}
    for feed in FEEDS:
        msg = fetch(session, feed)
        if msg is None:
            counts[feed] = None
            continue
        ts, n = msg.header.timestamp, 0
        for e in msg.entity:
            if e.HasField('trip_update'):
                tu = e.trip_update
                t = tu.trip
                for i, u in enumerate(tu.stop_time_update):
                    preds.writerow([now, feed, ts, t.trip_id, t.start_date, t.route_id, i, u.stop_id,
                                    u.arrival.time if u.HasField('arrival') else '',
                                    u.departure.time if u.HasField('departure') else ''])
                    n += 1
            elif e.HasField('vehicle'):
                v = e.vehicle
                vehs.writerow([now, feed, ts, v.trip.trip_id, v.trip.start_date, v.trip.route_id, v.stop_id,
                               v.current_status, v.current_stop_sequence, v.timestamp])
        counts[feed] = n
    files.flush()
    ok = sum(c is not None for c in counts.values())
    logging.info('poll %d: %d/8 feeds, %d prediction rows', now, ok, sum(c or 0 for c in counts.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw'))
    ap.add_argument('--interval', type=int, default=30)
    ap.add_argument('--polls', type=int, default=0, help='stop after this many polls (0 = run forever)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.out, 'logger.log'), level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    files, session, done = DayFiles(args.out), requests.Session(), [False]
    signal.signal(signal.SIGTERM, lambda *_: done.__setitem__(0, True))   # systemd stop: finish the poll, close cleanly
    logging.info('logger started, every %d s, output %s', args.interval, args.out)
    n = 0
    try:
        while not done[0]:
            start = time.time()
            try:
                poll_once(session, files)
            except Exception:
                logging.exception('poll failed; continuing')
            n += 1
            if args.polls and n >= args.polls:
                break
            time.sleep(max(0.0, args.interval - (time.time() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        files.close()
        logging.info('logger stopped after %d polls', n)


if __name__ == '__main__':
    main()
