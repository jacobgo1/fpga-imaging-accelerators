# Software

Host utilities, bare-metal applications and embedded Linux components, once a
board and software stack are chosen. Take AXI-Lite register offsets from the
driver headers generated inside the exported HLS IP rather than hardcoding them.

`fpga.sh` in the repository root wraps everything below in short commands
(`source fpga.sh; fpga_help`), and [docs/framework.md](../docs/framework.md)
shows how it fits together.

- `pynq/<kernel>/`: a Jupyter notebook (and optionally a driver module) for
  the board running PYNQ. Every bitstream build packages it with the build's
  files (below).

## On the board with PYNQ

Every successful `bitstream` build writes, next to the bitstream:

```text
build/<kernel>/<run>/<kernel>_pynq/      everything the notebook needs, in one place:
    <kernel>.bit  <kernel>.hwh             the design (PYNQ wants the same name for both)
    x<kernel>_hw.h                         register offsets
    <kernel>_pynq.py  <kernel>.ipynb       driver and notebook, from software/pynq/<kernel>/
    ...                                    files listed in software/pynq/<kernel>/include.txt
build/<kernel>/<run>/<kernel>_pynq.zip   the same, as one file to upload
```

To give a kernel a notebook, put it in `software/pynq/<kernel>/` (like
`software/pynq/matmul/matmul.ipynb`) and commit it: every build of that kernel
then carries a copy next to its bitstream, ready to run after uploading.
Re-package an existing build to pick up an edited notebook:
`python3 main.py pynq --kernel matmul` (or `fpga_pynq matmul`).

Upload the zip in Jupyter (`http://BOARD:9090`), run `!unzip -o justoliunet_pynq.zip`
in a notebook cell, open `justoliunet_pynq/justoliunet.ipynb` and run it from
the top: it loads the design, runs the self-test, classifies single pixels and
whole images, and shows the pictures. Images are prepared on a PC
(`tools/justoliunet_image.py prepare`, then `fpga_zip`) and uploaded next to the notebook.

## A trained network: `justoliunet`

`src/hls/justoliunet/` classifies one HYPSO-2 pixel as cloud (0), land (1) or
sea (2) from its **raw L1a spectrum** (120 bands). It does the whole chain
that training used (hypso-onboard-segmentation):

1. **preprocess**: keep 110 bands (drop 0-7 and 118-119), then z-score each
   with the training set's mean and std;
2. **1D-Justo-LiuNet** with the weights from `weights/fp32/justoliunet/`:
   4 x [Conv1D k=6 -> ReLU -> MaxPool 2], flatten, Dense -> 3 logits.

It works in float32, so it should reproduce the trained model. Band
selection, statistics and weights all come from one generated header:

```bash
python tools/export_justoliunet.py --mu-sd <prepared dataset>/mu_sd.txt   # numpy, not torch
python tools/export_justoliunet.py --placeholder-normalization            # until you have it
```

`mu_sd.txt` is what `dataset_processing/prepare_hypso_dataset.py` wrote into
the prepared dataset folder the checkpoints were trained on; the committed
kernel uses it. `--placeholder-normalization` (mean 0, std 1) builds and tests
without it, but classifies raw captures wrongly. The exporter also writes raw
test spectra and their expected logits for the testbench and the board. The
`justoliunet_bn` checkpoint works too.

```bash
source fpga.sh
fpga_test justoliunet      # C++ vs reference, on the PC
fpga_build justoliunet     # bitstream + build/justoliunet/latest/justoliunet_pynq.zip
```

On the board, the notebook in that zip loads the design and runs the
self-test: every pixel in `tb/data/justoliunet_vectors.txt`, PASS or FAIL.

### Classifying an image

An image is classified pixel by pixel. On a PC, `tools/justoliunet_image.py`
(numpy) turns a capture into pixels for the board plus the results the board
should give; the notebook runs them through the FPGA and scores them:

```bash
python3 tools/justoliunet_image.py prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy --step 8 --out aegean
fpga_zip aegean            # upload aegean.zip next to the notebook, then run section 4
```

The image can be a raw HYPSO-2 capture `.nc` (read like training does; needs
`pip install hypso`, and its labels are remapped as in training), a raw
`(H, W, 120)` `.npy`, or a prepared training image `data<i>.npy` (110 bands,
already z-scored) with `label<i>.npy`. A prepared image is turned back into
raw with the kernel's own statistics.

`compare` checks every pixel against a numpy reference of the exact kernel,
reports accuracy and IoU per class against the labels, and writes
`overview.png`, `fpga_classes.png`, `mismatch.png`, `labels.png` and the
input pictures into the image folder. It runs in the notebook, or on a PC
with `python3 tools/justoliunet_image.py compare aegean` once `fpga_logits.txt` is back. Each pixel is a
separate kernel call: `--crop ROW COL H W` a region, or `--step N` for a
subsampled view of the whole scene.
