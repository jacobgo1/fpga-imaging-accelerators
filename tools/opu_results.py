#!/usr/bin/env python3
"""The OPU boards' results, on the PC: classify (software/opu/) only runs the kernel and
times it on the board, so the classes are checked against the labels here.

    python tools/opu_results.py RESULTS --images FOLDER
        RESULTS: a folder benchmark.sh wrote (times.csv, scores/KERNEL/IMAGE.npy); FOLDER holds
        the image folders (IMAGE/labels.npy). Writes RESULTS/report.md: each kernel's times and
        accuracy and IoU, as the ZCU104's benchmark.ipynb.

    python tools/opu_results.py --scores scores.npy --labels labels.npy
        one run's accuracy and IoU.

Labels: 0 cloud, 1 land, 2 sea; other values are unlabeled and skipped.
"""
import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

CLASSES = ('cloud', 'land', 'sea')
PATCH = 32


def counts(scores, labels):
    """[labelled, correct, both cloud/land/sea, either cloud/land/sea] for one run: the
    labelled pixels, those classified as labelled, and per class the pixels both call it and
    either calls it (IoU = both / either)."""
    classes = scores[..., 3]
    if classes.shape != labels.shape:
        raise SystemExit(f'the scores are {classes.shape[0]} x {classes.shape[1]}, the labels {labels.shape}')
    known = (labels >= 0) & (labels <= 2)
    c, l = classes[known], labels[known]
    return np.array([known.sum(), (c == l).sum()]
                    + [((c == k) & (l == k)).sum() for k in range(3)]
                    + [((c == k) | (l == k)).sum() for k in range(3)], dtype=np.int64)


def iou_text(total):
    """Each class's IoU and the mIoU over the classes that occur, as table cells."""
    ious = [total[2 + k] / total[5 + k] if total[5 + k] else None for k in range(3)]
    seen = [v for v in ious if v is not None]
    cells = ['-' if v is None else f'{v:.3f}' for v in ious]
    return cells + [f'{sum(seen) / len(seen):.3f}' if seen else '-']


def check(args):
    total = counts(np.load(args.scores), np.load(args.labels))
    print(f'accuracy: {100 * total[1] / max(total[0], 1):.2f}% of {total[0]} labelled pixels')
    print('IoU ' + ', '.join(f'{name} {cell}' for name, cell in zip(CLASSES + ('mean',), iou_text(total))))


def per_unit(kernel, h, w):
    """The 1D networks' cycles are counted per pixel, the 2D ones' per 32 x 32 patch."""
    if kernel.startswith(('justoliunet', 'spectralpointwise1d')):
        return 'pixel', h * w
    return 'patch', math.ceil(h / PATCH) * math.ceil(w / PATCH)


def report(args):
    results, images = Path(args.results), Path(args.images)
    with open(results / 'times.csv', newline='') as f:
        runs = list(csv.DictReader(f))
    kernels, shapes, rows = {}, {}, []
    for run in runs:
        k, image = run['kernel'], run['image']
        kernels.setdefault(k, [])
        if run['error']:
            rows.append(f'| {image} | `{k}` | failed: {run["error"]} | | | | |')
            continue
        h, w, mhz = int(run['height']), int(run['width']), float(run['mhz'])
        shapes.setdefault(image, f'{image} ({h} x {w})')
        total = counts(np.load(results / 'scores' / k / f'{image}.npy'), np.load(images / image / 'labels.npy'))
        unit, units = per_unit(k, h, w)
        cycles = int(run['cycles']) if run['cycles'] else None
        own = cycles / (mhz * 1000) if cycles is not None else None
        kernels[k].append(dict(mhz=mhz, h=h, w=w, unit=unit, units=units, cycles=cycles, own=own, total=total,
                               fpga=float(run['fpga_ms']), load=float(run['load_s']), wall=float(run['wall_s'])))
        rows.append(f'| {image} | `{k}` | {"-" if own is None else f"{own:.3f}"} | {float(run["fpga_ms"]):.3f} '
                    f'| {float(run["wall_s"]):.2f} | {"-" if cycles is None else f"{cycles / units:.2f} a {unit}"} '
                    f'| {100 * total[1] / max(total[0], 1):.2f}% |')

    lines = [f'# Optimized kernels on the OPU board, {results.name}', '',
             f'Images: {", ".join(shapes.values())}.', '', '## Mean over the images', '',
             '| Kernel | MHz | Images | Cycles (own count) | Own count (ms) | FPGA (ms) | Cube into memory (s) '
             '| Wall clock (s) | ns a pixel | Accuracy (mean) | Accuracy (all pixels) | IoU cloud | IoU land | IoU sea '
             '| mIoU |', '|' + ' --- |' * 15]
    for k, done in kernels.items():
        if not done:
            continue
        n, counted = len(done), [r for r in done if r['cycles'] is not None]
        total = sum(r['total'] for r in done)
        mean = lambda key, rs=done: sum(r[key] for r in rs) / len(rs)
        cycles = f'{sum(r["cycles"] / r["units"] for r in counted) / len(counted):.2f} a {done[0]["unit"]}' if counted else '-'
        own = f'{mean("own", counted):.3f}' if counted else '-'
        ns = (f'{sum(1e3 * r["cycles"] / r["mhz"] / (r["h"] * r["w"]) for r in counted) / len(counted):.2f}'
              if counted else '-')
        accuracy = sum(r['total'][1] / max(r['total'][0], 1) for r in done) / n
        lines.append(f'| `{k}` | {done[0]["mhz"]:.1f} | {n} | {cycles} | {own} | {mean("fpga"):.3f} | {mean("load"):.2f} '
                     f'| {mean("wall"):.2f} | {ns} | {100 * accuracy:.2f}% | {100 * total[1] / max(total[0], 1):.2f}% | '
                     + ' | '.join(iou_text(total)) + ' |')
    lines += ['', '## Every run', '', '| Image | Kernel | Own count (ms) | FPGA (ms) | Wall clock (s) | Cycles | Accuracy |',
              '| --- | --- | --- | --- | --- | --- | --- |'] + rows
    text = '\n'.join(lines) + '\n'
    (results / 'report.md').write_text(text, encoding='utf-8', newline='\n')
    print(text + f'written: {results / "report.md"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', nargs='?', help="a folder benchmark.sh wrote")
    parser.add_argument('--images', help='the folder that holds the image folders (IMAGE/labels.npy)')
    parser.add_argument('--scores', help='one run: the scores.npy classify wrote')
    parser.add_argument('--labels', help='one run: its labels.npy')
    args = parser.parse_args()
    if args.scores and args.labels and not args.results:
        check(args)
    elif args.results and args.images and not args.scores:
        report(args)
    else:
        parser.error('give RESULTS --images FOLDER, or --scores and --labels')
    return 0


if __name__ == '__main__':
    sys.exit(main())
