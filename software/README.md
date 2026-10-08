# Software

What runs on the board's ARM: one Jupyter notebook per kernel, for PYNQ. Take
AXI-Lite register offsets from the driver header generated inside the exported
HLS IP (`x<kernel>_hw.h`, packed next to the notebook) rather than hardcoding
them.

`fpga.sh` in the repository root wraps everything below in short commands
(`source fpga.sh; fpga_help`), and [docs/framework.md](../docs/framework.md)
shows how it fits together.

## On the board with PYNQ

Every successful `bitstream` build writes, next to the bitstream:

```text
build/<kernel>/<run>/<kernel>_pynq/      everything the notebook needs, in one place:
    <kernel>.bit  <kernel>.hwh             the design (PYNQ wants the same name for both)
    x<kernel>_hw.h                         register offsets
    <kernel>.ipynb                         the notebook, from software/pynq/<kernel>/
    ...                                    files listed in software/pynq/<kernel>/include.txt
build/<kernel>/<run>/<kernel>_pynq.zip   the same, as one file to upload
```

To give a kernel a notebook, put it in `software/pynq/<kernel>/` and commit it:
every build of that kernel then carries a copy next to its bitstream. Re-package
an existing build to pick up an edited notebook: `fpga_pynq <kernel>`.

Upload the zip in Jupyter (`http://BOARD:9090`), run `!unzip -o <kernel>_pynq.zip`
in a notebook cell, open `<kernel>_pynq/<kernel>.ipynb` and run it from the top.

## The network: 2D-JustoUNet-Simple

Cloud (0), land (1) or sea (2) for every pixel of a HYPSO-2 capture, from
32 x 32 patches of the raw L1a cube (120 bands), with the int8 weights
`tools/quantize_weights.py` makes from `weights/fp32/justounetsimple/`.

| Kernel | What it is |
| --- | --- |
| `justounetsimple` | The golden baseline: one patch per start, float, plain loops. The notebook does the preprocessing (keep 110 bands, z-score with `mu_sd.txt`) and the patching. |
| `justounetsimple_opt` | The fast one: one start for the whole capture. It reads the raw cube from DDR and does the band selection, z-score, patching and all four blocks in hardware, in integer fixed point. |

```bash
source fpga.sh
python3 tools/quantize_weights.py weights/fp32/justounetsimple   # once: weights/quantized/ is not in git
fpga_test justounetsimple_opt      # Python tests + the C++ testbench, on the PC
fpga_build justounetsimple_opt     # bitstream + build/justounetsimple_opt/latest/justounetsimple_opt_pynq.zip
```

### An image for the notebook

On a PC, `tools/justounetsimple_image.py` (numpy) turns a capture into what the
notebook classifies, plus the answer the board should give:

```bash
python3 tools/justounetsimple_image.py prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy --out aegean_unet
fpga_zip aegean_unet               # upload aegean_unet.zip next to the notebook
```

It writes `cube.npy` (the raw cube, uint16), `reference_scores.npy` (the numpy
model's scores for the same patches), `labels.npy` and `meta.json`. Reading a
`.nc` capture needs `pip install hypso`, as in training; a raw `(H, W, 120)`
`.npy` works too. `--crop ROW COL HEIGHT WIDTH` takes a region only. The same
image folder works for both notebooks.

The `justounetsimple_opt` notebook loads the cube into DDR, self-tests on the
first patch, classifies the whole image in one start and prints the wall-clock
and FPGA times, checks the scores against the reference and the labels
(accuracy, IoU per class), and draws the picture, the classes, the labels and
where they differ.
