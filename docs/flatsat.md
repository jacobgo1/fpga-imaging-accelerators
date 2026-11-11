# Testing a design on the HYPSO-2 flatsat

How to run a `_zynq7030` kernel on the flatsat, the ground copy of HYPSO-2: its OPU
(PicoBoB, a Zynq-7030) with the camera, on the CAN bus to `ws2`. The kernel runs on a real
capture (`binned_cube.bip`), and the class of every pixel comes back to the PC. How the
tarball is built and what is in it: [opu.md](opu.md).

Every command below uses `sp_unet_small_opt_u8_zynq7030`, the first kernel run this way
(2026-11-10). For another kernel, change that name everywhere; nothing else changes.

## How it differs from the ZedBoard

| | ZedBoard ([opu.md](opu.md)) | Flatsat |
| --- | --- | --- |
| Reached by | SSH (`zb05`) | `hypso-cli` on `ws2`, over CAN (1 Mbit/s: uploads are slow) |
| Commands on the OPU | `ssh zb05 ...` | `shell remote oneshot 12 TIMEOUT_S COMMAND` |
| Files to the OPU | `scp` | `opu upload` |
| Load the design | `K/load.sh` | `obip fpga reload K.bit.bin -o K.dtbo` |
| Linux | development image, nothing else running | flight software running (opu-services): **do not capture while the design is loaded** |
| Image | a test image (`binned_cube.bip`, `capture_config.ini`), its labels scored on the PC (`tools/image.py score`) | a capture's `binned_cube.bip`, no true labels: times and labels only |

## What stays where on the OPU

| What | Where |
| --- | --- |
| The tarball, unpacked | `/media/sd-pl/` (`firmware/`, `K/`) |
| Captures | `/media/sd-pl/hsiN/` (opu-services' working folder is `/media/sd-pl/`) |
| The OPU's boot files | `/media/sd-img/`: **write nothing here** |
| Your design loaded on the FPGA | until `obip fpga reset` or a reboot |

## 1. Build, on the lab server

From `~/Documents/MartinOgJacob/fpga-imaging-accelerators`:

```sh
source fpga.sh
fpga_build sp_unet_small_opt_u8_zynq7030 picobob
```

Only cnn_accelerator changed: `fpga_package sp_unet_small_opt_u8_zynq7030 picobob`. The result is
`build/sp_unet_small_opt_u8_zynq7030/picobob/sp_unet_small_opt_u8_zynq7030_opu.tar.gz`.

## 2. Copy it to ws2, from the lab server

Still on the server, in the same folder (`ws2` is in the server's `~/.ssh/config`; it asks for
the `hypso` password):

```sh
scp build/sp_unet_small_opt_u8_zynq7030/picobob/sp_unet_small_opt_u8_zynq7030_opu.tar.gz ws2:MartinOgJacob/
```

It lands in `/home/hypso/MartinOgJacob/` on `ws2`.

## 3. Start hypso-cli and the OPU, on ws2

```sh
cd /home/hypso/martin
rlwrap ./hypso-cli 15 -ccan0=0/0
```

`RTNETLINK answers: Operation not permitted` at the start is harmless: hypso-cli tries to set
up `can0`, which only root may do, and it is already up.

In hypso-cli, power the OPU on, wait until it answers, and give remote commands 300 s
instead of 10:

```
shell remote oneshot 4 3 output 10 1 0
csp ping 12
shell remote timeout 12 300
```

Repeat `csp ping 12` until it answers (a minute or two). If it never does, check that the EPS
took the command: `shell remote oneshot 4 3 hkout`. To watch the OPU boot, in a second `ws2`
terminal: `picocom -b 115200 /dev/ttyUSB0` (`sudo killall picocom` if it says busy).

## 4. Upload and unpack

```
opu upload /home/hypso/MartinOgJacob/sp_unet_small_opt_u8_zynq7030_opu.tar.gz /media/sd-pl/sp_unet_small_opt_u8_zynq7030_opu.tar.gz
shell remote oneshot 12 30 tar -xzf /media/sd-pl/sp_unet_small_opt_u8_zynq7030_opu.tar.gz -C /media/sd-pl/
shell remote oneshot 12 10 ls -l /media/sd-pl/firmware/ /media/sd-pl/sp_unet_small_opt_u8_zynq7030/
```

Full paths: `opu upload` does not expand `~`. `cnn_accelerator` is built with each kernel's
own driver: every kernel brings its own, in its own folder.

## 5. A capture

Use one already there, or take one. Both while the OPU's own design is loaded (before step 6).

```
shell remote oneshot 12 10 ls -l /media/sd-pl/
```

Take a new one (100 frames; `-d b` saves `binned_cube.bip`, `-k` skips the compression):

```
hsi capture -n 100 -d b -k
```

It goes in the next free `/media/sd-pl/hsiN`. `cnn_accelerator` reads its size from the capture's
`capture_config.ini`. To look at it yourself:

```
shell remote oneshot 12 10 ls -l /media/sd-pl/hsi1
shell remote oneshot 12 10 cat /media/sd-pl/hsi1/capture_config.ini
```

The bands must be 120 (`column_count = 1080`, `bin_factor = 9`, `sample_divisor = 1`).
`hsi1` on 2026-11-10: 598 x 1092, 156 723 840 bytes.

## 6. Load, run, restore

```
obip fpga reload sp_unet_small_opt_u8_zynq7030.bit.bin -o sp_unet_small_opt_u8_zynq7030.dtbo
shell remote oneshot 12 10 cat /sys/class/uio/uio*/name
shell remote oneshot 12 300 /media/sd-pl/sp_unet_small_opt_u8_zynq7030/process-capture.sh hsi1
obip fpga reset
```

- The `uio` names must include `hls_kernel` (the overlay's registers); otherwise do not run.
- `obip fpga reset` puts the OPU's own design back. **Always run it**, before any capture.

`process-capture.sh` writes `/media/sd-pl/hsi1/sp_unet_small_opt_u8_zynq7030-labels.bin`
(a byte a pixel, HYPSO's values: sea 0, land 1, cloud 2, in the cube's order, as the OPU's
own 1D CNN writes `cnn-1d-labels.bin`) and its log to `/media/sd-pl/hsi1/logs/`:

```
shell remote oneshot 12 10 cat /media/sd-pl/hsi1/logs/sp_unet_small_opt_u8_zynq7030.log
```

Check the clock once per design, in the log: the kernel's own count of cycles over the ARM's
FPGA time should be the clock it was built for. On 2026-11-10 (the earlier `classify.c`):
598 x 1092, FPGA 180.7 ms, 18 066 885 cycles, so 100.0 MHz; the cube from the SD card into
memory took 6.7 s, 7.1 s in all. `obip fpga reload` has no fallback when the overlay cannot
set the clock, unlike `load.sh`.

## 7. The labels back to the PC

In hypso-cli:

```
opu download /media/sd-pl/hsi1/sp_unet_small_opt_u8_zynq7030-labels.bin /home/hypso/MartinOgJacob/labels.bin
```

Then on the laptop (598 1092: the capture's height and width, from step 5):

```sh
scp ws2:/home/hypso/MartinOgJacob/labels.bin .
python -c "import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.colors import ListedColormap; c = np.fromfile('labels.bin', dtype=np.uint8).reshape(598, 1092); plt.imshow(c, cmap=ListedColormap(['blue', 'green', 'white']), vmin=0, vmax=2, interpolation='nearest'); plt.title('sea (blue) / land (green) / cloud (white)'); plt.savefig('labels.png', dpi=120, bbox_inches='tight')"
start labels.png
```

It writes `labels.png` rather than opening a window: from Git Bash, `plt.show()`'s window can
open behind the terminal, which then seems to hang until the window is closed.

## 8. Switch the OPU off

```
opu shutdown
```

and confirm with `y`. The power goes off 40 s later: **do not power it on again before then**
(`shell remote oneshot 4 3 output 10 1 0` within the 40 s leaves it powered but never
answering). If that happens, power cycle it:

```
shell remote oneshot 4 3 output 10 0 0
shell remote oneshot 4 3 output 10 1 0
```

## When something goes wrong

| What you see | Why | What to do |
| --- | --- | --- |
| `csp ping 12` fails | the OPU is off or still booting | step 3: power it on, wait, ping again |
| a remote command stops after 10 s | the OPU's default command timeout | `shell remote timeout 12 300` |
| `K initialize failed: Device not found` | the design or its overlay is not loaded | `obip fpga reload K.bit.bin -o K.dtbo`, then the `uio` check |
| the cube's size does not match | the cube is not what `capture_config.ini` says (not 120 bands, or cut short) | read `capture_config.ini` and `ls -l` the capture (step 5) |
| the kernel does not finish | the design is not running (clock, wrong bitstream) | check the clock (step 6); `obip fpga reset` |

More, for the build and cnn_accelerator: [opu.md](opu.md), "When something goes wrong".
