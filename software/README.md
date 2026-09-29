# Software

Host utilities, bare-metal applications and embedded Linux components, once a
board and software stack are chosen. Take AXI-Lite register offsets from the
driver headers generated inside the exported HLS IP rather than hardcoding them.

`fpga.sh` in the repository root wraps everything below in short commands
(`source fpga.sh; fpga_help`), and [docs/framework.md](../docs/framework.md)
shows how it fits together.

- `pynq/<kernel>/`: **the normal way.** A driver and a Jupyter notebook for
  the board running PYNQ. Every bitstream build packages them with the build's
  files (below).
- `jtag/`: fallback without Linux on the board: the PC drives the kernel over
  the JTAG cable with `xsdb`.

## On the board with PYNQ (`pynq/`)

Every successful `bitstream` build writes, next to the bitstream:

```text
artifacts/<kernel>/<build>/pynq/          everything the notebook needs, in one place:
    <kernel>.bit  <kernel>.hwh             the design (PYNQ wants the same name for both)
    x<kernel>_hw.h                         register offsets
    <kernel>_pynq.py  <kernel>.ipynb       driver and notebook, from software/pynq/<kernel>/
    ...                                    files listed in software/pynq/<kernel>/include.txt
artifacts/<kernel>/<build>/<kernel>_pynq.zip   the same, as one file to upload
```

For builds made before this existed: `python3 main.py pynq --kernel justoliunet`
(or `fpga_pynq`), which takes the `.hwh` from the build's `.xsa`.

Upload the zip in Jupyter (`http://BOARD:9090`), run `!unzip -o justoliunet_pynq.zip`
in a notebook cell, open `justoliunet_pynq/justoliunet.ipynb` and run it from
the top: it loads the design, runs the self-test, classifies single pixels and
whole images, and shows the pictures. Images are prepared on a PC
(`fpga_prepare`, then `fpga_zip`) and uploaded next to the notebook.

## Talking to a kernel over JTAG (`jtag/`, fallback)

For when the board runs no Linux. With no software on the board at all,
`xsdb`, the debugger that ships with Vivado, reads and writes
the kernel's AXI-Lite registers through the JTAG cable, via the Zynq PS.

This only works for kernels whose arrays are on the `s_axilite` bundle
(`matmul` is), because then every element is a register. Streaming (`axis`) or
`bram` ports need a DMA in the block design first.

On the machine the board is plugged into, with the Vivado settings sourced
and the board in JTAG boot mode:

```bash
python3 main.py bitstream --kernel matmul        # with board_script = boards/zcu104/system.tcl
xsdb software/jtag/matmul.tcl artifacts/matmul/<run>
```

That programs the bitstream, initializes the PS (clock, resets, PS-to-PL
ports), and runs a self-test against a software reference. No separate Vivado
Hardware Manager step is needed; the script resets the board and programs it
itself.

To type in your own matrices, add `-live`. The board is programmed once, then
you are asked for A and B repeatedly; each pair runs on the FPGA and the
result is printed with MATCH or MISMATCH against a software reference:

```bash
xsdb software/jtag/matmul.tcl artifacts/matmul/<run> -live
```

```text
A> random -1000 1000          random values in a range (default -16..16)
B> 1 2 3 4 5 6 7 8            64 numbers, row by row, over one or more lines
  8/64> 0 1 0 0 0 0 0 0
  ...
A> same                       reuse the previous A (also: identity, zero)
B> file my_b.txt              64 numbers from a text file
A> quit
```

From the `xsdb` prompt the same pieces are procs:

```tcl
source software/jtag/matmul.tcl
board_open artifacts/matmul/<run>
mm_check [mm_load my_a.txt] [mm_random]     ;# run, print, compare
mm_live                                     ;# the prompt above
mm_selftest 20
kernel_write a {...64 values...}; kernel_run; kernel_read c -signed   ;# raw access
kernel_status
```

`board.tcl` is kernel-independent: it takes array offsets, widths and depths
from the HLS `*_hw.h` header that `boards/zcu104/system.tcl` copies into
`artifacts/<kernel>/<run>/board/`, next to `psu_init.tcl` and `address.tcl`
(the kernel's base address). If `address.tcl` is missing, pass the base from
the Vivado Address Editor: `board_open <dir> 0xA0000000`.

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
fpga_test                  # C++ vs reference, on the PC
fpga_build                 # bitstream + artifacts/justoliunet/<build>/justoliunet_pynq.zip
```

On the board, the notebook in that zip loads the design and runs the
self-test: every pixel in `tb/data/justoliunet_vectors.txt`, PASS or FAIL.

### Classifying an image

An image is classified pixel by pixel. On a PC, `tools/justoliunet_image.py`
(numpy) turns a capture into pixels for the board plus the results the board
should give; the notebook runs them through the FPGA and scores them:

```bash
fpga_prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy --step 8 --out aegean
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
with `fpga_compare aegean` once `fpga_logits.txt` is back. Each pixel is a
separate kernel call: `--crop ROW COL H W` a region, or `--step N` for a
subsampled view of the whole scene.

Without Linux on the board, the same runs over JTAG: `fpga_selftest`,
`fpga_shell` and `fpga_classify aegean` (boot switches on JTAG).
