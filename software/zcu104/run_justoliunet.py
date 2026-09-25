#!/usr/bin/env python3
"""Run the justoliunet HLS kernel on a PYNQ-loaded ZCU104 bitstream.

On-board script: needs `pynq` and `numpy`, not `torch`. Convert a trained
model's weights to the .npz this expects with export_weights.py on your
dev machine first (where torch is available), then copy bitstream + .hwh +
.npz to the board and run this there.

Usage:
    python3 run_justoliunet.py system.bit weights.npz --input tile.npy
    python3 run_justoliunet.py system.bit weights.npz          # random test input

Geometry (H, W, K, BASE_CH, NUM_CLASSES below) must match
src/hls/justoliunet/justoliunet.hpp's JNET_* macros for whatever bitstream
you're loading -- if you retarget one, retarget both and rebuild.
"""
import argparse
import sys
import time

import numpy as np

try:
    from pynq import Overlay, allocate
except ImportError:
    sys.exit("This script runs on the board, inside PYNQ's Python "
              "environment -- it needs the 'pynq' package, which only a "
              "PYNQ image provides.")

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
WEIGHT_KEYS = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "w5", "b5")
POINTER_ORDER = ("din", "w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "w5", "b5", "dout")


def find_register(register_map, base_name):
    """Vitis HLS's exact register name for a pointer argument varies by
    tool version (seen: 'din', 'din_1', 'din_V'). Try the likely spellings
    instead of hardcoding one, and fail with the real list if none match."""
    candidates = [base_name, f"{base_name}_1", f"{base_name}_V", f"{base_name}_r", f"{base_name}_V_1"]
    for candidate in candidates:
        if hasattr(register_map, candidate):
            return candidate
    available = sorted(name for name in dir(register_map) if not name.startswith("_"))
    raise AttributeError(
        f"No register found for argument {base_name!r}; tried {candidates}.\n"
        f"Registers this IP actually has: {available}\n"
        "Add the real name to find_register()'s candidate list above.")


def load_weights(npz_path):
    data = np.load(npz_path)
    missing = [key for key in WEIGHT_KEYS if key not in data]
    if missing:
        sys.exit(f"{npz_path} is missing arrays: {missing} "
                  "(see export_weights.py, or --keys to remap your checkpoint's names)")
    weights = {}
    for key in WEIGHT_KEYS:
        array = np.asarray(data[key], dtype=np.float32)
        if array.shape != SHAPES[key]:
            sys.exit(f"{key}: expected shape {SHAPES[key]}, got {array.shape} in {npz_path}")
        weights[key] = array
    return weights


def load_input(path):
    if path is None:
        print("No --input given; using a fixed random test tile.")
        rng = np.random.default_rng(1)
        return rng.uniform(-1.0, 1.0, size=SHAPES["din"]).astype(np.float32)
    array = np.asarray(np.load(path), dtype=np.float32)
    if array.shape == (H, W):
        array = array[:, :, None]
    if array.shape != SHAPES["din"]:
        sys.exit(f"--input: expected shape {SHAPES['din']} (or {(H, W)}), got {array.shape}")
    return array


def find_ip(overlay, ip_name):
    if ip_name:
        return getattr(overlay, ip_name)
    candidates = [name for name in overlay.ip_dict if "justoliunet" in name.lower()]
    if len(candidates) != 1:
        sys.exit(f"Could not pick one IP automatically; overlay.ip_dict has: "
                  f"{list(overlay.ip_dict)}. Pass --ip-name explicitly.")
    print(f"Using IP instance: {candidates[0]}")
    return getattr(overlay, candidates[0])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bitstream", help="e.g. system.bit (its matching .hwh must sit next to it)")
    parser.add_argument("weights", help=".npz with w1,b1,w2,b2,w3,b3,w4,b4,w5,b5 -- see export_weights.py")
    parser.add_argument("--input", help=f".npy tile, shape {SHAPES['din']} or {(H, W)}; "
                                        "random test data if omitted")
    parser.add_argument("--ip-name", help="IP instance name in the overlay; auto-detected if omitted")
    parser.add_argument("--output", help="save dout to this .npy instead of just printing it")
    parser.add_argument("--timeout", type=float, default=10.0,
                         help="seconds to wait for AP_DONE before giving up (default: 10)")
    args = parser.parse_args()

    print(f"Loading overlay {args.bitstream} ...")
    overlay = Overlay(args.bitstream)
    ip = find_ip(overlay, args.ip_name)

    weights = load_weights(args.weights)
    din_data = load_input(args.input)

    print("Allocating DMA buffers ...")
    buffers = {name: allocate(shape, dtype=np.float32) for name, shape in SHAPES.items()}
    buffers["din"][:] = din_data
    for key, array in weights.items():
        buffers[key][:] = array

    print("Pointing kernel registers at the buffers ...")
    for name in POINTER_ORDER:
        register = find_register(ip.register_map, name)
        setattr(ip.register_map, register, buffers[name].physical_address)

    print("Starting kernel ...")
    ip.register_map.CTRL.AP_START = 1
    started = time.time()
    while not ip.register_map.CTRL.AP_DONE:
        if time.time() - started > args.timeout:
            sys.exit(f"Timed out after {args.timeout}s waiting for AP_DONE -- check "
                      "boards/zcu104/system.tcl's wiring and the CTRL register name.")
    elapsed_ms = (time.time() - started) * 1000

    result = np.array(buffers["dout"])
    print(f"Done in {elapsed_ms:.2f} ms (wall clock, includes Python/register overhead).")
    print("Result (dout):")
    print(result)
    print("Predicted class per row:", result.argmax(axis=1).tolist())

    if args.output:
        np.save(args.output, result)
        print(f"Saved to {args.output}")

    for buffer in buffers.values():
        buffer.freebuffer()


if __name__ == "__main__":
    main()
