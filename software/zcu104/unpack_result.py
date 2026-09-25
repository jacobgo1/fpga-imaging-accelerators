#!/usr/bin/env python3
"""Decode jtag_run.tcl's dout.bin (raw little-endian float32, H*NUM_CLASSES
values) back into something readable.

Usage: python3 unpack_result.py dout.bin
"""
import sys

import numpy as np

H, NUM_CLASSES = 4, 4


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: unpack_result.py dout.bin")
    result = np.fromfile(sys.argv[1], dtype="<f4")
    if result.size != H * NUM_CLASSES:
        sys.exit(f"{sys.argv[1]}: expected {H * NUM_CLASSES} floats, got {result.size} "
                  "-- H/NUM_CLASSES here no longer match the kernel's geometry?")
    result = result.reshape(H, NUM_CLASSES)
    print(result)
    print("Predicted class per row:", result.argmax(axis=1).tolist())


if __name__ == "__main__":
    main()
