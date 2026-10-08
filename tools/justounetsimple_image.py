#!/usr/bin/env python3
"""Make an image folder for the justounetsimple notebook on the board (numpy).

    python tools/justounetsimple_image.py prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy \\
        --out aegean_unet [--crop ROW COL HEIGHT WIDTH]

Writes, into --out:
    cube.npy               the raw L1a cube (H x W x 120; uint16 when the values are raw
                           counts), what the board classifies
    reference_scores.npy   the numpy model's scores (H x W x 3) for the same 32 x 32 patches,
                           which the notebook checks the FPGA against
    labels.npy             the labels, remapped as in training (if --labels)
    meta.json              where it came from, and the RGB bands for the picture

The patches are made exactly as the notebook makes them: the image is grown to whole
32 x 32 patches by repeating its last row and column. Classes: 0 cloud, 1 land, 2 sea.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import justounetsimple_model as model
import patching

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


def reference_scores(cube, kept, mean, inv_std, layers):
    """The numpy model over the same patches the board gets."""
    scores = np.zeros(cube.shape[:2] + (3,), dtype=np.float32)
    total = patching.count(cube.shape)
    start = time.monotonic()
    for n, (patch, tile) in enumerate(patching.iter_patches(cube, pad_mode='edge'), start=1):
        patching.place(scores, model.forward_patch(model.preprocess(patch, kept, mean, inv_std), layers), tile)
        if n % 100 == 0 or n == total:
            print(f'  reference: {n}/{total} patches, {time.monotonic() - start:.0f} s', flush=True)
    return scores


def load_cube(path):
    if str(path).endswith('.nc'):
        return load_capture(path), capture_rgb(path)
    cube = np.load(path)
    if cube.ndim == 3 and cube.shape[-1] != model.RAW_BANDS and cube.shape[0] == model.RAW_BANDS:
        cube = cube.transpose(1, 2, 0)
    if cube.ndim != 3 or cube.shape[-1] != model.RAW_BANDS:
        raise SystemExit(f'cube has shape {cube.shape}; expected (H, W, {model.RAW_BANDS}) raw bands')
    return cube.astype(np.float32), None


def prepare(args):
    kept, mean, inv_std = model.read_preprocessing(args.mu_sd)
    layers = model.load_weights(args.weights)
    cube, rgb_default = load_cube(args.cube)
    print(f'cube: {cube.shape[0]} x {cube.shape[1]} pixels, {cube.shape[2]} bands')
    row, col, height, width = args.crop or (0, 0, cube.shape[0], cube.shape[1])
    region = (slice(row, row + height), slice(col, col + width))
    labels = None
    if args.labels:
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
        cube = cube.astype(np.uint16)    # raw counts, as captured: what justounetsimple_opt reads

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'cube.npy', cube)
    if labels is not None:
        np.save(out / 'labels.npy', labels)
    print(f'{cube.shape[0]} x {cube.shape[1]} pixels, {patching.count(cube.shape)} patches; '
          'running the numpy model for the reference ...')
    np.save(out / 'reference_scores.npy', reference_scores(cube, kept, mean, inv_std, layers))
    (out / 'meta.json').write_text(json.dumps({
        'cube': str(args.cube), 'crop': [row, col, height, width],
        'rgb_bands': args.rgb or rgb_default or DEFAULT_RGB}, indent=2))
    print(f'Written to {out}. Zip it (fpga_zip {out.as_posix()}) and upload it next to the notebook.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='capture -> raw cube for the board + reference scores')
    p.add_argument('cube', help='CAPTURE-l1a.nc, or a raw .npy cube (H, W, 120) or (120, H, W)')
    p.add_argument('--labels', help='(H, W) .npy: the capture\'s _labels.npy')
    p.add_argument('--out', required=True, help='directory for this image')
    p.add_argument('--crop', type=int, nargs=4, metavar=('ROW', 'COL', 'HEIGHT', 'WIDTH'),
                   help='a region only (a full capture is 665 patches)')
    p.add_argument('--rgb', type=int, nargs=3, metavar=('R', 'G', 'B'), help='raw band indices for the picture')
    p.add_argument('--mu-sd', default=model.MU_SD)
    p.add_argument('--weights', default=model.WEIGHTS)
    args = parser.parse_args()
    prepare(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
