#!/usr/bin/env python3
"""Export a trained JustoLiuNet PyTorch checkpoint to the .npz
run_justoliunet.py loads on the board.

Dev-machine script: needs torch. run_justoliunet.py itself only needs
numpy, so this is the one place torch has to be installed.

Usage:
    python3 export_weights.py model.pt weights.npz
    python3 export_weights.py model.pt weights.npz \\
        --keys layers.0.weight=w1 layers.0.bias=b1     # remap non-default names

Default state_dict key -> justoliunet argument mapping matches the
nn.Module in src/golden/justoliunet_golden.hpp's docstring:
  self.conv1/pool1, conv2/pool2, conv3/pool3, conv4/pool4, self.fc
If your model used different attribute names, override with --keys instead
of renaming your checkpoint.
"""
import argparse
import sys

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("Needs torch (pip install torch) -- run this on your dev machine, "
              "not the board; the .npz it writes only needs numpy to read back.")

DEFAULT_KEY_MAP = {
    "conv1.weight": "w1", "conv1.bias": "b1",
    "conv2.weight": "w2", "conv2.bias": "b2",
    "conv3.weight": "w3", "conv3.bias": "b3",
    "conv4.weight": "w4", "conv4.bias": "b4",
    "fc.weight": "w5", "fc.bias": "b5",
}


def state_dict_from(checkpoint):
    """Unwrap the common checkpoint shapes: a bare state_dict, a dict with
    one nested under 'state_dict'/'model', or a saved nn.Module itself."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
        return checkpoint["model"].state_dict()
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help=".pt/.pth file")
    parser.add_argument("output", help="output .npz path")
    parser.add_argument("--keys", nargs="*", default=[],
                         help="override state_dict_key=arg_name pairs, e.g. layers.0.weight=w1")
    args = parser.parse_args()

    key_map = dict(DEFAULT_KEY_MAP)
    for pair in args.keys:
        if "=" not in pair:
            sys.exit(f"--keys entries must look like state_dict_key=arg_name, got: {pair!r}")
        state_key, arg_name = pair.split("=", 1)
        key_map[state_key] = arg_name

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = state_dict_from(checkpoint)

    arrays, missing = {}, []
    for state_key, arg_name in key_map.items():
        if state_key not in state_dict:
            missing.append(state_key)
            continue
        arrays[arg_name] = state_dict[state_key].detach().cpu().numpy().astype("float32")

    if missing:
        sys.exit(f"Checkpoint is missing keys: {missing}\n"
                  f"Available keys: {sorted(state_dict.keys())}\n"
                  "Remap them with --keys real.name.weight=w1 ...")

    print("Exporting:")
    for arg_name, array in sorted(arrays.items()):
        print(f"  {arg_name:4} {tuple(array.shape)}")

    np.savez(args.output, **arrays)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
