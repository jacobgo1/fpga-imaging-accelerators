"""Cut an image into 32 x 32 patches for the justounetsimple kernel, and stitch the answers back.

The kernel takes one fixed-size patch (HWC, float32) and returns one score map
of the same height and width. For a whole capture the host:

    patches, tiles = cut(image)            # image: H x W x C
    scores = [run_kernel(p) for p in patches]
    result = stitch(scores, tiles)         # H x W x classes

The image is padded so it divides into whole patches. With halo=0 the patches
do not overlap, which matches how the model was trained (isolated 32 x 32
patches, zero padding inside the convolutions). A pixel on a patch edge then
sees zeros instead of its neighbour. With halo=h each patch also covers h
pixels of its neighbours on every side, and only the middle (patch - 2*h) is
kept, so those edge pixels get real context. The price is more patches.
"""
import numpy as np

PATCH = 32


def _layout(height, width, patch, halo):
    if halo < 0 or 2 * halo >= patch:
        raise ValueError(f'halo must be 0 <= halo < patch/2, got {halo} for patch {patch}')
    step = patch - 2 * halo
    rows = -(-height // step)   # ceiling division
    cols = -(-width // step)
    return step, rows, cols


def cut(image, patch=PATCH, halo=0, pad_mode='reflect'):
    """Return (patches, tiles). patches: N x patch x patch x C. tiles: where each belongs."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError('image must be H x W x C')
    height, width, _ = image.shape
    step, rows, cols = _layout(height, width, patch, halo)
    # Pad so the kept centres of the patches cover the image exactly: halo on
    # the top and left, and enough on the bottom and right to fill whole steps.
    bottom = rows * step + 2 * halo - height - halo
    right = cols * step + 2 * halo - width - halo
    # 'reflect' needs less padding than the image is wide; fall back to edge
    # replication for tiny images.
    mode = pad_mode
    if mode == 'reflect' and (max(halo, bottom) >= height or max(halo, right) >= width):
        mode = 'edge'
    padded = np.pad(image, ((halo, bottom), (halo, right), (0, 0)), mode=mode)
    patches, tiles = [], []
    for r in range(rows):
        for c in range(cols):
            y, x = r * step, c * step
            patches.append(padded[y:y + patch, x:x + patch])
            tiles.append((r * step, c * step))   # top-left of the kept centre, in image pixels
    return np.stack(patches), tiles


def stitch(results, tiles, shape, patch=PATCH, halo=0):
    """Put per-patch results back into an H x W x classes map; shape is the image's (H, W, ...)."""
    height, width = shape[0], shape[1]
    results = np.asarray(results)
    step = patch - 2 * halo
    out = np.zeros((height, width, results.shape[-1]), dtype=results.dtype)
    for result, (y, x) in zip(results, tiles):
        centre = result[halo:halo + step, halo:halo + step]
        h = min(step, height - y)
        w = min(step, width - x)
        out[y:y + h, x:x + w] = centre[:h, :w]
    return out
