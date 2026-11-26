"""Salvage recorder files damaged by a crash, power cut or hard reboot.

The logger appends one gzip member per run. If the machine dies mid-write, the file holds a broken stretch and
ordinary readers stop there, losing the rest of the day even though later runs appended good data after it.
This reads member by member, skips each broken stretch to the next gzip header, drops the half-written line,
and keeps only rows with the header's column count and a numeric poll field. It streams, so memory stays small.

Usage:  python repair.py data/raw/2026-11-08            repair both files of a day in place (originals to data/damaged/)
        python repair.py --check data/raw data/vm       list damaged files without changing anything
        python repair.py --check data/vm/2026-11-20     (copy folders or day folders)
"""
import argparse, glob, gzip, mmap, os, sys, zlib

PIECE = 1 << 16


def is_damaged(path):
    try:
        with gzip.open(path, 'rb') as f:
            while f.read(1 << 24):
                pass
        return False
    except (OSError, EOFError, zlib.error):
        return True


def salvage(src, dst):
    """Copy every readable row of src into dst. Returns (rows kept, number of damaged stretches skipped)."""
    with open(src, 'rb') as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data, \
            gzip.open(dst, 'wb', compresslevel=6) as out:
        header = ncomma = None
        pos, tail, kept, faults = 0, b'', 0, 0

        def emit(lines):
            nonlocal header, ncomma, kept
            for line in lines:
                if header is None:
                    header, ncomma = line, line.count(b',')
                    out.write(line + b'\n')
                elif line != header and line.count(b',') == ncomma and line.split(b',', 1)[0].isdigit():
                    out.write(line + b'\n')
                    kept += 1

        while pos < len(data):
            d, i = zlib.decompressobj(31), pos
            try:
                while i < len(data) and not d.eof:
                    lines = (tail + d.decompress(data[i:i + PIECE])).split(b'\n')
                    tail = lines.pop()
                    emit(lines)
                    i += PIECE
                if not d.eof:
                    raise zlib.error('member ends early')
                pos = i - len(d.unused_data)
            except zlib.error:
                faults += 1
                tail = b''                                      # the line being written when it broke
                nxt = data.find(b'\x1f\x8b\x08', max(i - PIECE, pos) + 1)
                if nxt == -1:
                    break
                pos = nxt
        emit([tail] if tail else [])
    return kept, faults


def repair_day(folder):
    """Originals go to data/damaged/<copy>-<day>/, outside the day folders, so weekly archives don't pick them up."""
    folder = os.path.abspath(folder)
    copy_dir, day = os.path.split(folder)
    keep = os.path.join(os.path.dirname(copy_dir), 'damaged', f'{os.path.basename(copy_dir)}-{day}')
    for name in ('predictions', 'vehicles'):
        path = os.path.join(folder, f'{name}.csv.gz')
        if not os.path.exists(path) or not is_damaged(path):
            continue
        tmp = path + '.tmp'
        kept, faults = salvage(path, tmp)
        os.makedirs(keep, exist_ok=True)
        os.replace(path, os.path.join(keep, f'{name}.csv.gz'))
        os.replace(tmp, path)
        print(f'{path}: kept {kept:,} rows, skipped {faults} damaged stretch(es); original moved to {keep}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--check', action='store_true', help='only list damaged files')
    args = ap.parse_args()
    if args.check:
        files = sorted(f for p in args.paths                  # a copy folder (data/raw) or one day folder
                       for f in glob.glob(os.path.join(p, '*', '*.csv.gz')) + glob.glob(os.path.join(p, '*.csv.gz')))
        bad = [f for f in files if is_damaged(f)]
        print('\n'.join(bad) if bad else f'all {len(files)} files OK')
        sys.exit(1 if bad else 0)
    for folder in args.paths:
        repair_day(folder)


if __name__ == '__main__':
    main()
