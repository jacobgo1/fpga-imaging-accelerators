# Testing a design on the HYPSO-2 flatsat

How to run a `_zynq7030` kernel on the flatsat, the ground copy of HYPSO-2: its OPU
(PicoBoB, a Zynq-7030) with the camera, on the CAN bus to `ws2`. The kernel runs on a real
capture (`binned_cube.bip`), and the class of every pixel comes back to the PC.

Every command below uses `sp_unet_small_opt_u8_zynq7030`, the first kernel run this way
(2026-11-10). For another kernel, change that name everywhere; nothing else changes.

## How it differs from the ZedBoard

| | ZedBoard ([zedboard.md](zedboard.md)) | Flatsat |
| --- | --- | --- |
| Reached by | SSH (`zb05`) | `hypso-cli` on `ws2`, over CAN (1 Mbit/s: uploads are slow) |
| Commands on the OPU | `ssh zb05 ...` | `shell remote oneshot 12 TIMEOUT_S COMMAND` |
| Files to the OPU | `scp` | `obip fpga upload` (bitstream, md5sum, overlay), `opu upload` (anything else) |
| Load the design | `load.sh` | `obip fpga reload K.bit.bin` (takes `K.dtbo` beside it) |
| Linux | development image, nothing else running | flight software running (opu-services): **do not capture while the design is loaded** |
| Image | a test image's `cube.bip` (`tools/image.py`); its labels checked on the PC | a capture's `binned_cube.bip`, no labels: times and classes only |

## What stays where on the OPU

| What | Where |
| --- | --- |
| Bitstreams, overlays, `classify` | `/media/sd-pl/firmware/` |
| Captures | `/media/sd-pl/hsiN/` (opu-services' working folder is `/media/sd-pl/`) |
| The OPU's boot files | `/media/sd-img/`: **write nothing here** |
| Your design loaded on the FPGA | until `obip fpga reset` or a reboot |

## 1. Build or repack, on the lab server

From `~/Documents/MartinOgJacob/fpga-imaging-accelerators`, after `source fpga.sh`. A kernel
with no bitstream yet:

```sh
fpga_build sp_unet_small_opt_u8_zynq7030 --board zynq7030
```

A kernel that has one, when only `classify.c` changed (repacks the newest bitstream run):

```sh
python3 main.py pynq --kernel sp_unet_small_opt_u8_zynq7030 --board zynq7030
```

The packed folder must hold `K.bit.bin`, `K.bit.bin.md5sum`, `K.dtbo` and `classify`:

```sh
ls "$(ls -d build/sp_unet_small_opt_u8_zynq7030/*bitstream* | tail -n 1)"/sp_unet_small_opt_u8_zynq7030_opu
```

No `.dtbo` or no `classify`: the server lacks `dtc` or the ARM compiler
(`sudo apt install device-tree-compiler gcc-arm-linux-gnueabihf`); install them and repack.

## 2. Copy it to ws2, from the lab server

Still on the server, in the same folder (`ws2` is in the server's `~/.ssh/config`; it asks for
the `hypso` password):

```sh
scp -r "$(ls -d build/sp_unet_small_opt_u8_zynq7030/*bitstream* | tail -n 1)"/sp_unet_small_opt_u8_zynq7030_opu ws2:MartinOgJacob/
```

It lands in `/home/hypso/MartinOgJacob/sp_unet_small_opt_u8_zynq7030_opu` on `ws2`.

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

## 4. Upload the design

```
obip fpga upload /home/hypso/MartinOgJacob/sp_unet_small_opt_u8_zynq7030_opu
opu upload /home/hypso/MartinOgJacob/sp_unet_small_opt_u8_zynq7030_opu/classify /media/sd-pl/firmware/classify
shell remote oneshot 12 10 chmod +x /media/sd-pl/firmware/classify
shell remote oneshot 12 10 ls -l /media/sd-pl/firmware/
```

Full paths: `opu upload` does not expand `~`. `obip fpga upload` sends only the `.bit.bin`,
`.md5sum` and `.dtbo` files, hence the second upload. `classify` is the same program for every
optimized kernel, but it is built with each kernel's registers: upload it again with every
new kernel.

## 5. A capture

Use one already there, or take one. Both while the OPU's own design is loaded (before step 6).

```
shell remote oneshot 12 10 ls -l /media/sd-pl/
```

Take a new one (100 frames; `-d b` saves `binned_cube.bip`, `-k` skips the compression):

```
hsi capture -n 100 -d b -k
```

It goes in the next free `/media/sd-pl/hsiN`. Read its size:

```
shell remote oneshot 12 10 ls -l /media/sd-pl/hsi1
shell remote oneshot 12 10 cat /media/sd-pl/hsi1/capture_config.ini
```

HEIGHT is `frame_count`; WIDTH is `row_count / spatial_subsampling`. The bands must be 120
(`column_count = 1080`, `bin_factor = 9`, `sample_divisor = 1`), and `binned_cube.bip` exactly
HEIGHT x WIDTH x 240 bytes (`classify` checks it). `hsi1` on 2026-11-10: 598 x 1092,
156 723 840 bytes.

## 6. Load, run, restore

```
obip fpga reload sp_unet_small_opt_u8_zynq7030.bit.bin
shell remote oneshot 12 10 cat /sys/class/uio/uio*/name
shell remote oneshot 12 300 /media/sd-pl/firmware/classify /media/sd-pl/hsi1/binned_cube.bip /media/sd-pl/hsi1/scores.npy 598 1092
obip fpga reset
```

- The `uio` names must include `kernel` (the overlay's registers); otherwise do not run.
- `obip fpga reset` puts the OPU's own design back. **Always run it**, before any capture.
- Do not run `obip fpga hash` while your design is loaded: it reads a memory that is only in
  the OPU's own design.

What `classify` prints, on 2026-11-10:

```
598 x 1092 pixels, 665 patches, in 1 start of up to 598 rows
cube into memory:      6.738 s
FPGA:                  180.7 ms, 271.7 us per patch
own count:          18066885 cycles
scores to the file:    0.158 s
wall clock:            7.096 s
```

Check the clock: own count / FPGA time should be the kernel's clock (here 18 066 885 / 0.1807 s
= 100.0 MHz). `obip fpga reload` has no fallback when the overlay cannot set it, unlike
`load.sh`. A kernel without its own count: read the clock with
`shell remote oneshot 12 10 devmem 0xF8000108; devmem 0xF8000170` (`load.sh` shows how they
give the clock).

## 7. The classes back to the PC

`scores.npy` is H x W x 4 int32: the cloud, land and sea scores, then the class (0 cloud,
1 land, 2 sea). 10 MB for 598 x 1092, slow over CAN. In hypso-cli:

```
opu download /media/sd-pl/hsi1/scores.npy /home/hypso/MartinOgJacob/scores.npy
```

Then on the laptop:

```sh
scp ws2:/home/hypso/MartinOgJacob/scores.npy .
python -c "import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.colors import ListedColormap; c = np.load('scores.npy')[..., 3]; plt.imshow(c, cmap=ListedColormap(['white', 'green', 'blue']), vmin=0, vmax=2, interpolation='nearest'); plt.title('cloud (white) / land (green) / sea (blue)'); plt.savefig('classes.png', dpi=120, bbox_inches='tight')"
start classes.png
```

It writes `classes.png` rather than opening a window: from Git Bash, `plt.show()`'s window can
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
| `no UIO device named kernel` from `classify` | the design or its overlay is not loaded | `obip fpga reload K.bit.bin`, then the `uio` check |
| `binned_cube.bip is N bytes, but ...` | HEIGHT or WIDTH is not the capture's | read `capture_config.ini` again (step 5) |
| `the kernel did not finish` | the design is not running (clock, wrong bitstream) | check the clock (step 6); `obip fpga reset` |
| `classify` is missing after `obip fpga upload` | that command sends only bitstreams and overlays | `opu upload` it (step 4) |
