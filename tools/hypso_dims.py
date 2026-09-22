#!/usr/bin/env python3
"""Report the real dimensions of HYPSO hyperspectral captures.

Host-side only: uses the `hypso` package (pip install hypso), not any AMD
tool, and never touches build/ or reports/. The one thing it answers is what
src/hls/conv2d/conv2d.hpp needs before it can be retargeted at real captures:
how many spectral bands and how many spatial pixels an actual HYPSO-1/HYPSO-2
file has, as int16, so CONV_IC/CONV_IH/CONV_IW can be set from data instead
of guessed.

Usage:
    python tools/hypso_dims.py CAPTURE-l1a.nc [more.nc ...]
    python tools/hypso_dims.py --dir /path/to/captures
    python tools/hypso_dims.py --dir /path/to/captures --json dims.json

HYPSO capture filenames encode the product level the hypso package needs to
pick a loader: <target>_<UTC timestamp>-<level>.nc, for example
aegean_2024-08-22T08-41-46Z-l1a.nc. Point this at whatever L1a/L1b/L1c/L1d
files you already have; it never downloads anything.
"""
import argparse
import json
import sys
from pathlib import Path

try:
    from hypso import Hypso1
except ImportError:
    sys.exit("The 'hypso' package is not installed in this environment.\n"
              "    pip install hypso\n"
              "This is host-side only; it has nothing to do with the AMD toolchain "
              "or main.py's dependency-free promise.")

# _load_capture_file (hypso/HypsoBase.py) stores the cube under one of these
# attribute names depending on the product level parsed from the filename.
CUBE_ATTR = {'l1a': 'l1a_cube', 'l1b': 'l1b_cube', 'l1c': 'l1b_cube', 'l1d': 'l1d_cube'}


def describe(path: Path) -> dict:
    """One capture file's geometry, read via the public Hypso1 API."""
    capture = Hypso1(path, verbose=False)
    level = str(getattr(capture, 'product_level', '')).lower()

    # spatial_dimensions is (frame_count, image_height) in the library's own
    # terms -- frame_count is the along-track line count built up over the
    # capture, image_height is actually the across-track sample count per
    # line. Neither is a "height" in the row/col sense, so report both
    # plainly rather than repeating the library's confusing names.
    lines, samples = (int(d) for d in capture.spatial_dimensions)
    bands = int(capture.bands)

    info = {
        'file': str(path),
        'capture_name': getattr(capture, 'capture_name', path.stem),
        'product_level': level or 'unknown',
        'lines': lines,      # along-track
        'samples': samples,  # across-track
        'bands': bands,      # spectral channels after binning
        'wavelength_range_nm': None,
        'cube_shape': None,
        'value_range': None,
    }

    wavelengths = getattr(capture, 'wavelengths', None)
    if wavelengths is not None and len(wavelengths):
        info['wavelength_range_nm'] = [float(min(wavelengths)), float(max(wavelengths))]

    cube = getattr(capture, CUBE_ATTR.get(level, ''), None)
    if cube is not None:
        info['cube_shape'] = tuple(int(d) for d in cube.shape)
        values = cube.to_numpy() if hasattr(cube, 'to_numpy') else cube
        info['value_range'] = [float(values.min()), float(values.max())]

    info['pixels'] = lines * samples * bands
    info['bytes_as_int16'] = info['pixels'] * 2
    return info


def format_bytes(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.1f} {unit}' if unit != 'B' else f'{n} {unit}'
        n /= 1024
    return f'{n:.1f} GB'


def print_report(info: dict) -> None:
    print(f"{info['capture_name']}  ({info['product_level']})")
    print(f"  file      {info['file']}")
    print(f"  lines     {info['lines']}   (along-track)")
    print(f"  samples   {info['samples']}   (across-track)")
    print(f"  bands     {info['bands']}   (spectral channels)")
    if info['wavelength_range_nm']:
        low, high = info['wavelength_range_nm']
        print(f"  spectrum  {low:.1f} - {high:.1f} nm")
    if info['cube_shape']:
        print(f"  cube      shape {info['cube_shape']}  dtype loaded as float64 "
              f"(on-disk values are raw sensor counts)")
    if info['value_range']:
        low, high = info['value_range']
        print(f"  values    {low:g} .. {high:g}")
    print(f"  as int16  {format_bytes(info['bytes_as_int16'])} for the full cube "
          f"(lines x samples x bands x 2 bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*', type=Path, help='HYPSO .nc capture files')
    parser.add_argument('--dir', type=Path, help='Directory to glob *.nc files from')
    parser.add_argument('--json', type=Path, help='Also write the parsed results as JSON')
    args = parser.parse_args()

    files = list(args.files)
    if args.dir:
        files += sorted(args.dir.glob('*.nc'))
    if not files:
        parser.error('Give one or more .nc files, or --dir a directory containing them.')

    results = []
    for path in files:
        if not path.is_file():
            print(f'skip: not a file: {path}', file=sys.stderr)
            continue
        try:
            info = describe(path)
        except Exception as error:  # a malformed/unexpected capture must not abort the batch
            print(f'skip: could not read {path}: {error}', file=sys.stderr)
            continue
        results.append(info)
        print_report(info)
        print()

    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        print(f'Wrote {args.json}')

    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
