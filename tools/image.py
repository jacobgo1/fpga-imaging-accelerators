#!/usr/bin/env python3
"""Make an image folder for the boards (numpy), and score what an OPU board made of it.

    python tools/image.py prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy \\
        --out aegean_unet [--crop ROW COL HEIGHT WIDTH]
    python tools/image.py score LABELS.bin IMAGE_DIR
    python tools/image.py score RESULTS_DIR --images DIR

prepare writes, into --out:
    cube.npy               the raw L1a cube (H x W x 120; uint16 when the values are raw
                           counts), what the PYNQ notebooks classify
    binned_cube.bip        the same cube as HYPSO stores a capture, and
    capture_config.ini     its size (frame_count = H, row_count = W): what cnn_accelerator
                           reads on the OPU boards (only for a uint16 cube)
    labels.npy             the labels, remapped as in training: what the board is checked against
    meta.json              where it came from, and the RGB bands for the picture

score: cnn_accelerator's labels file (H x W uint8, HYPSO's values: sea 0, land 1, cloud 2)
against IMAGE_DIR/labels.npy: accuracy and IoU per class, as the notebooks count them. For a
folder from software/opu/scripts/benchmark.sh (KERNEL/IMAGE-labels.bin), every file in it,
each against --images/IMAGE/labels.npy, as a table.
Classes (labels.npy): 0 cloud, 1 land, 2 sea.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocessing  # noqa: E402

# Label remapping from hypso-onboard-segmentation (prepare_hypso_dataset.py remap_labels):
# 0 cloud, 1 land, 2 sea; other values are unlabeled.
RAW_LABEL_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
DEFAULT_RGB = [56, 67, 86]  # HYPSO-2 r/g/b bands, as in the capture -meta.json files


def load_capture(path):
    """A raw HYPSO-2 capture's L1a cube, read as prepare_hypso_dataset.py does."""
    try:
        from hypso import Hypso2
    except ImportError:
        raise SystemExit('reading .nc captures needs the hypso package, as in training: pip install hypso')
    return Hypso2(path=Path(path)).l1a_cube.to_numpy().astype(np.float32)


def capture_rgb(nc_path):
    """The capture's own RGB band choice, from the -meta.json beside it."""
    for meta_path in Path(nc_path).parent.glob('*-meta.json'):
        meta = json.loads(meta_path.read_text())
        if all(f'{c}_band' in meta for c in 'rgb'):
            return [int(meta[f'{c}_band']) for c in 'rgb']
    return None


def remap_labels(raw):
    labels = raw.astype(np.int64)
    for value, cls in RAW_LABEL_TO_CLASS.items():
        labels[raw == value] = cls
    return labels


def load_cube(path):
    if str(path).endswith('.nc'):
        return load_capture(path), capture_rgb(path)
    cube = np.load(path)
    bands = preprocessing.RAW_BANDS
    if cube.ndim == 3 and cube.shape[-1] != bands and cube.shape[0] == bands:
        cube = cube.transpose(1, 2, 0)
    if cube.ndim != 3 or cube.shape[-1] != bands:
        raise SystemExit(f'cube has shape {cube.shape}; expected (H, W, {bands}) raw bands')
    return cube.astype(np.float32), None


def prepare(args):
    cube, rgb_default = load_cube(args.cube)
    print(f'cube: {cube.shape[0]} x {cube.shape[1]} pixels, {cube.shape[2]} bands')
    row, col, height, width = args.crop or (0, 0, cube.shape[0], cube.shape[1])
    region = (slice(row, row + height), slice(col, col + width))
    labels = np.load(args.labels)
    if labels.shape != cube.shape[:2]:
        raise SystemExit(f'labels have shape {labels.shape}, cube is {cube.shape[:2]}')
    if str(args.cube).endswith('.nc'):
        labels = remap_labels(labels)
    labels = labels[region]
    cube = np.ascontiguousarray(cube[region])
    if cube.size == 0:
        raise SystemExit('the crop is empty')
    if not np.isfinite(cube).all():
        raise SystemExit('the cube contains NaN or infinite values')
    if cube.min() >= 0 and cube.max() <= 65535 and np.array_equal(cube, np.rint(cube)):
        cube = cube.astype(np.uint16)    # raw counts, as captured: what the optimized kernels read

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'cube.npy', cube)
    if cube.dtype == np.uint16:
        # As a capture is stored, frame by frame (the cube's rows), each pixel's bands together
        cube.tofile(out / 'binned_cube.bip')
        (out / 'capture_config.ini').write_text(
            f'frame_count = {cube.shape[0]}\nrow_count = {cube.shape[1]}\naoi_y = 0\nexposure = 0.0\n')
    else:
        print('not raw counts (uint16): no binned_cube.bip for the OPU boards')
    np.save(out / 'labels.npy', labels)
    (out / 'meta.json').write_text(json.dumps({
        'cube': str(args.cube), 'crop': [row, col, height, width],
        'rgb_bands': args.rgb or rgb_default or DEFAULT_RGB}, indent=2))
    print(f'Written to {out}: {cube.shape[0]} x {cube.shape[1]} pixels. Zip it (fpga_zip '
          f"{out.as_posix()}) and upload it to the folder above the designs' folders, once for all of them.")


HYPSO_TO_CLASS = np.array([2, 1, 0], dtype=np.int64)   # sea 0, land 1, cloud 2 -> 2, 1, 0
CLASS_NAMES = ('cloud', 'land', 'sea')


def score_one(labels_bin, image_dir):
    """Counts for one labels file against the image's labels: labelled pixels, correct ones,
    and per class the pixels both call it and either calls it."""
    truth = np.load(Path(image_dir) / 'labels.npy')
    hypso = np.fromfile(labels_bin, dtype=np.uint8)
    if hypso.size != truth.size:
        raise SystemExit(f'{labels_bin}: {hypso.size} labels, {image_dir}/labels.npy has {truth.size}')
    if hypso.max(initial=0) > 2:
        raise SystemExit(f'{labels_bin}: values above 2, not a labels file')
    classes = HYPSO_TO_CLASS[hypso].reshape(truth.shape)
    known = (truth >= 0) & (truth < 3)
    c, t = classes[known], truth[known]
    both = [int(((c == k) & (t == k)).sum()) for k in range(3)]
    either = [int(((c == k) | (t == k)).sum()) for k in range(3)]
    return int(known.sum()), int((c == t).sum()), both, either


def describe(counts):
    labelled, correct, both, either = counts
    accuracy = 100 * correct / labelled if labelled else 0
    iou = [b / e if e else None for b, e in zip(both, either)]
    present = [v for v in iou if v is not None]
    return accuracy, iou, sum(present) / len(present) if present else None


def score(args):
    path = Path(args.labels)
    if path.is_file():
        if not args.image:
            raise SystemExit('score LABELS.bin IMAGE_DIR: the image folder with labels.npy')
        counts = score_one(path, args.image)
        accuracy, iou, miou = describe(counts)
        print(f'accuracy: {accuracy:.2f}% of {counts[0]} labelled pixels')
        print('IoU: ' + ', '.join(f'{n} {"-" if v is None else f"{v:.3f}"}' for n, v in zip(CLASS_NAMES, iou))
              + ('' if miou is None else f', mIoU {miou:.3f}'))
        return
    images = Path(args.images or '.')
    runs = sorted(path.glob('*/*-labels.bin'))
    if not runs:
        raise SystemExit(f'no KERNEL/IMAGE-labels.bin in {path}')
    print('| Kernel | Image | Accuracy | IoU cloud | IoU land | IoU sea | mIoU |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    for run in runs:
        image = run.name[:-len('-labels.bin')]
        accuracy, iou, miou = describe(score_one(run, images / image))
        cells = ' | '.join('-' if v is None else f'{v:.3f}' for v in iou)
        print(f'| `{run.parent.name}` | {image} | {accuracy:.2f}% | {cells} | {"-" if miou is None else f"{miou:.3f}"} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='capture -> raw cube for the board, labels and the picture bands')
    p.add_argument('cube', help='CAPTURE-l1a.nc, or a raw .npy cube (H, W, 120) or (120, H, W)')
    p.add_argument('--labels', required=True,
                   help='(H, W) .npy: the capture\'s _labels.npy; the board is checked against them')
    p.add_argument('--out', required=True, help='directory for this image')
    p.add_argument('--crop', type=int, nargs=4, metavar=('ROW', 'COL', 'HEIGHT', 'WIDTH'),
                   help='a region only (a full capture is 665 patches)')
    p.add_argument('--rgb', type=int, nargs=3, metavar=('R', 'G', 'B'), help='raw band indices for the picture')
    p = sub.add_parser('score', help="an OPU board's labels file against the image's labels")
    p.add_argument('labels', help="cnn_accelerator's labels file, or a benchmark.sh results folder")
    p.add_argument('image', nargs='?', help='the image folder with labels.npy (for one labels file)')
    p.add_argument('--images', help='for a results folder: where the image folders are (default: here)')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else score)(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
