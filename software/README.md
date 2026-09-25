# Software

Host utilities, bare-metal applications and embedded Linux components, once a
board and software stack are chosen. Take AXI-Lite register offsets from the
driver headers generated inside the exported HLS IP rather than hardcoding them.

## Talking to a kernel over JTAG (`jtag/`)

The quickest way to put data into a kernel and read results back, with no
embedded software: `xsdb`, the debugger that ships with Vivado, reads and writes
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
the prepared dataset folder the checkpoints were trained on. With
`--placeholder-normalization` (mean 0, std 1; what is committed now) the
hardware is complete and passes every test, but raw captures come out
wrongly classified: re-export with `--mu-sd` and rebuild before judging real
data. The exporter also writes raw test spectra and their expected logits for
the testbench and the board. The `justoliunet_bn` checkpoint works too.

```bash
python3 main.py native --kernel justoliunet          # C++ vs reference
python3 main.py bitstream --kernel justoliunet --config config/project.local.json
xsdb software/jtag/justoliunet.tcl artifacts/justoliunet/<run>
```

The board test runs every pixel in `tb/data/justoliunet_vectors.txt` on the
FPGA and prints PASS or FAIL per pixel. Live, from the `xsdb` prompt:

```tcl
source software/jtag/justoliunet.tcl
board_open artifacts/justoliunet/<run>
jl_vector 4                     ;# one test pixel: FPGA vs reference logits
jl_classify_file my_pixel.txt   ;# 120 raw band values, any separators
```

### Classifying an image

An image is classified pixel by pixel. `tools/justoliunet_image.py` (numpy)
sends raw pixels to the board and scores what comes back:

```bash
python3 tools/justoliunet_image.py prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy \
    --step 8 --out img
xsdb software/jtag/justoliunet.tcl artifacts/justoliunet/<run> \
    -pixels img/pixels.txt img/fpga_logits.txt
python3 tools/justoliunet_image.py compare img
```

The image can be a raw HYPSO-2 capture `.nc` (read like training does; needs
`pip install hypso`, and its labels are remapped as in training), a raw
`(H, W, 120)` `.npy`, or a prepared training image `data<i>.npy` (110 bands,
already z-scored) with `label<i>.npy`. A prepared image is turned back into
raw with the kernel's own statistics, so it classifies correctly even with
placeholder normalization.

`compare` checks every pixel against a numpy reference of the exact kernel,
reports accuracy and IoU per class against the labels, and writes `rgb.png`,
`fpga_classes.png`, `reference_classes.png`, `mismatch.png` and `labels.png`
into `img/`. Over JTAG a pixel takes tens of milliseconds: `--crop ROW COL H W`
a region, or `--step N` for a subsampled view of the whole scene.
