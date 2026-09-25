#!/usr/bin/env python3
"""Pack justoliunet's weights + one input tile into a single raw float32
binary blob, laid back-to-back in argument order (din, w1, b1, ..., dout) --
what jtag_run.tcl downloads straight into DDR over JTAG with `dow -data`.
There's no OS on the board to read separate files from in the JTAG-only
flow, so everything goes in as one blob at one fixed address.

Usage:
    python3 pack_data_bin.py weights.npz --input tile.npy -o data.bin
    python3 pack_data_bin.py weights.npz -o data.bin          # random test tile

weights.npz comes from export_weights.py. Prints each array's byte offset
in the blob -- these must match the offsets baked into jtag_run.tcl (they
will, as long as H/W/K/BASE_CH/NUM_CLASSES below still match
src/hls/justoliunet/justoliunet.hpp's JNET_* macros).
"""
import argparse
import sys

import numpy as np

H, W, K, BASE_CH, NUM_CLASSES = 4, 110, 3, 8, 4


def conv_len(length, k):
    return length - k + 1


def pool_len(length):
    return length // 2


def flatten_dim(width, k, base_ch):
    length = width
    for _ in range(4):
        length = pool_len(conv_len(length, k))
    return 4 * base_ch * length


FLAT = flatten_dim(W, K, BASE_CH)
SHAPES = {
    "din": (H, W, 1),
    "w1": (BASE_CH, 1, K),                     "b1": (BASE_CH,),
    "w2": (2 * BASE_CH, BASE_CH, K),           "b2": (2 * BASE_CH,),
    "w3": (3 * BASE_CH, 2 * BASE_CH, K),       "b3": (3 * BASE_CH,),
    "w4": (4 * BASE_CH, 3 * BASE_CH, K),       "b4": (4 * BASE_CH,),
    "w5": (NUM_CLASSES, FLAT),                 "b5": (NUM_CLASSES,),
    "dout": (H, NUM_CLASSES),
}
ORDER = ("din", "w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "w5", "b5", "dout")
WEIGHT_KEYS = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "w5", "b5")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", help=".npz with w1,b1,...,b5 (see export_weights.py)")
    parser.add_argument("--input", help=f".npy tile, shape {SHAPES['din']} or {(H, W)}; "
                                        "random test data if omitted")
    parser.add_argument("-o", "--output", required=True, help="output .bin path")
    args = parser.parse_args()

    data = np.load(args.weights)
    arrays = {}
    for key in WEIGHT_KEYS:
        if key not in data:
            sys.exit(f"{args.weights} is missing '{key}' (see export_weights.py)")
        array = np.asarray(data[key], dtype="<f4")
        if array.shape != SHAPES[key]:
            sys.exit(f"{key}: expected shape {SHAPES[key]}, got {array.shape}")
        arrays[key] = array

    if args.input:
        din = np.asarray(np.load(args.input), dtype="<f4")
        if din.shape == (H, W):
            din = din[:, :, None]
    else:
        print("No --input given; using a fixed random test tile.")
        rng = np.random.default_rng(1)
        din = rng.uniform(-1.0, 1.0, size=SHAPES["din"]).astype("<f4")
    if din.shape != SHAPES["din"]:
        sys.exit(f"--input: expected shape {SHAPES['din']} (or {(H, W)}), got {din.shape}")
    arrays["din"] = din
    arrays["dout"] = np.zeros(SHAPES["dout"], dtype="<f4")   # the kernel fills this in on the board

    offset = 0
    print(f"{'array':6} {'bytes':>8} {'offset':>10}")
    with open(args.output, "wb") as f:
        for key in ORDER:
            raw = arrays[key].tobytes()
            f.write(raw)
            print(f"{key:6} {len(raw):8} {offset:10}")
            offset += len(raw)
    print(f"Total {offset} bytes -> {args.output}")


if __name__ == "__main__":
    main()
