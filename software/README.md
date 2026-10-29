# Software

What runs on the board's ARM: two Jupyter notebooks for PYNQ, shared by every
model, in `software/pynq/shared/`: `classify_golden.ipynb` for the golden kernels
(the kernel named like its model) and `classify.ipynb` for the optimized ones
(every `*_opt*`). Each takes the register offsets from the driver header generated
inside the exported HLS IP (`x<kernel>_hw.h`, packed next to the notebook) and
which model it is from `kernel.json` (main.py writes it from `tools/models.py`).
`software/opu/` holds the C programs for the boards running the OPU system's PetaLinux.

`fpga.sh` in the repository root wraps everything below in short commands
(`source fpga.sh; fpga_help`), and [docs/framework.md](../docs/framework.md)
shows how it fits together.

## On the board with PYNQ

Every successful `bitstream` build writes, next to the bitstream:

```text
build/<kernel>/<run>/<kernel>_pynq/      everything the notebook needs, in one place:
    <kernel>.bit  <kernel>.hwh             the design (PYNQ wants the same name for both)
    x<kernel>_hw.h                         register offsets
    classify.ipynb or classify_golden.ipynb  the notebook, from software/pynq/shared/
    kernel.json                            which model, its patch size and scores' format
    mu_sd.txt                              (golden only) the training mean/std per band
build/<kernel>/<run>/<kernel>_pynq.zip   the same, as one file to upload
```

A kernel of no model in `tools/models.py` takes its own
notebook from `software/pynq/<kernel>/`, and any files its `include.txt` names.
Re-package an existing build to pick up an edited notebook: `fpga_pynq <kernel>`.

Upload the zip in Jupyter (`http://BOARD:9090`), run `!unzip -o <kernel>_pynq.zip`
in a notebook cell, open the notebook in `<kernel>_pynq/` and run it from the top.

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

On a PC, `tools/image.py` (numpy) turns a capture into what the notebooks
classify, and `tools/models.py` adds the answers the board should give, for
every model:

```bash
python3 tools/image.py prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy --out aegean_unet
fpga_zip aegean_unet               # upload aegean_unet.zip above the designs' folders, once
```

The folder holds `cube.npy` (the raw cube, uint16), `labels.npy` and `meta.json`.
Reading a `.nc` capture needs `pip install hypso`, as in
training; a raw `(H, W, 120)` `.npy` works too. `--crop ROW COL HEIGHT WIDTH`
takes a region only (do, for the golden kernels: they are slow).

Both notebooks print the same: the wall-clock and FPGA times (per patch for a
2D network, per pixel for a 1D one; the optimized kernels also by their own cycle
counter, best of 5 too), accuracy and IoU per class against the labels, and the
picture, the classes, the labels and where they differ. `labels.npy` is required:
without it they stop before the FPGA runs. They do not compare with the PC's
models: tests/test_notebooks.py checks that against fake kernels.
