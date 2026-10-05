"""Cut an image into 32 x 32 patches for the justounetsimple kernel, and stitch the answers back.

The kernel takes one fixed-size patch (HWC, float32) and returns one score map
of the same height and width. For a whole capture the host:

    for patch, tile in iter_patches(image):    # image: H x W x C
        scores = run_kernel(patch)
        place(result, scores, tile)            # result: H x W x classes

(`cut` and `stitch` do the same with everything in memory at once.)

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


def count(shape, patch=PATCH, halo=0):
    """How many patches an image of this (H, W, ...) shape takes."""
    _, rows, cols = _layout(shape[0], shape[1], patch, halo)
    return rows * cols


def padded(image, patch=PATCH, halo=0, pad_mode='reflect'):
    """The image with halo pixels added on the top and left, and enough on the bottom and
    right that whole steps cover it."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError('image must be H x W x C')
    height, width, _ = image.shape
    step, rows, cols = _layout(height, width, patch, halo)
    bottom = rows * step + 2 * halo - height - halo
    right = cols * step + 2 * halo - width - halo
    # 'reflect' needs less padding than the image is wide; fall back to edge
    # replication for tiny images.
    mode = pad_mode
    if mode == 'reflect' and (max(halo, bottom) >= height or max(halo, right) >= width):
        mode = 'edge'
    return np.pad(image, ((halo, bottom), (halo, right), (0, 0)), mode=mode)


def iter_patches(image, patch=PATCH, halo=0, pad_mode='reflect'):
    """Yield (patch, tile) in raster order, one at a time. tile = (row, col) of the
    patch's kept centre in image pixels."""
    height, width = image.shape[0], image.shape[1]
    step, rows, cols = _layout(height, width, patch, halo)
    pad = padded(image, patch, halo, pad_mode)
    for r in range(rows):
        for c in range(cols):
            y, x = r * step, c * step
            yield pad[y:y + patch, x:x + patch], (y, x)


def cut(image, patch=PATCH, halo=0, pad_mode='reflect'):
    """Return (patches, tiles). patches: N x patch x patch x C. tiles: where each belongs."""
    patches, tiles = [], []
    for p, tile in iter_patches(image, patch, halo, pad_mode):
        patches.append(p)
        tiles.append(tile)
    return np.stack(patches), tiles


def place(out, result, tile, patch=PATCH, halo=0):
    """Write one patch's result (patch x patch x classes) into the H x W x classes map."""
    y, x = tile
    step = patch - 2 * halo
    centre = result[halo:halo + step, halo:halo + step]
    h = min(step, out.shape[0] - y)
    w = min(step, out.shape[1] - x)
    out[y:y + h, x:x + w] = centre[:h, :w]


def stitch(results, tiles, shape, patch=PATCH, halo=0):
    """Put per-patch results back into an H x W x classes map; shape is the image's (H, W, ...)."""
    results = np.asarray(results)
    out = np.zeros((shape[0], shape[1], results.shape[-1]), dtype=results.dtype)
    for result, tile in zip(results, tiles):
        place(out, result, tile, patch, halo)
    return out
