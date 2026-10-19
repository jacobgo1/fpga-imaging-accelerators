# Testing a design on the ZedBoard

What to do after `fpga_build K --board zedboard`: get the new bitstream onto the lab's
ZedBoard (`zb05`), load it, and check its answers on a test image.

The board already runs the OPU system's Linux (opu-system's PetaLinux) from its SD card.
Nothing about Linux has to be installed or uploaded: a design is loaded while it runs, with
`fpgautil`. (Only if the SD card's Linux ever has to be rebuilt:
[zedboard_linux_image.md](zedboard_linux_image.md).)

Everything below runs on the lab server, from
`~/Documents/MartinOgJacob/fpga-imaging-accelerators`.

## What stays where

| What | Where | Lasts |
| --- | --- | --- |
| The board's Linux | SD card | always: never upload it |
| Your design's files and the test image | `/media/sd-pl/firmware/` on the board (SD card) | until you overwrite them |
| Your design **loaded on the FPGA** | the FPGA | **until the next reboot**: then `load.sh` again |

At boot the board loads its own design (with UIO devices `cnn_1d_accelerator`,
`cnn_1d_memory`, `bitstream_hash`). Loading yours replaces it until the next reboot.

## Once: before the first test

**1. Tools on the server.** `fpga_build` uses them when it packs the design: `dtc` makes
the devicetree overlay (`.dtbo`), `arm-linux-gnueabihf-gcc` builds `classify` (the board
has neither).

```bash
sudo apt install device-tree-compiler gcc-arm-linux-gnueabihf
```

**2. The test image on the board.** `classify` takes a raw cube (H x W x 120, **uint16**)
and the PC's answer for it, `reference_fixed.npy`. They stay on the board; every design
uses the same two files.

The `aegean_unet/` on the server has its cube saved as float32 (it is older than the
uint16 format), which `classify` rejects. Its values are whole raw counts, and its
`reference_fixed.npy` was computed from them as uint16, so a uint16 copy matches it:

```bash
python3 -c "
import numpy as np
c = np.load('aegean_unet/cube.npy')
assert (c >= 0).all() and (c <= 65535).all() and (c == np.rint(c)).all(), 'not raw counts'
np.save('aegean_unet/cube_u16.npy', c.astype(np.uint16))
print('written aegean_unet/cube_u16.npy', c.shape)
"
ssh zb05 mkdir -p /media/sd-pl/firmware
scp aegean_unet/cube_u16.npy zb05:/media/sd-pl/firmware/cube.npy
scp aegean_unet/reference_fixed.npy zb05:/media/sd-pl/firmware/
```

A new test image comes from a HYPSO capture with
`python3 tools/justounetsimple_image.py prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy --out NAME`,
which writes `cube.npy` as uint16 and `reference_fixed.npy` (it says
`... and the integer model, for justounetsimple_opt ...` when it does). Copy those two the
same way.

## Every new design, step by step

### 1. Name the kernel

Once per terminal; every command below uses it:

```bash
K=justounetsimple_opt_zynq7020
```

The ZedBoard has the Zynq-7020 chip, so the kernel must be sized for it:
`justounetsimple_opt_zynq7020`, not `justounetsimple_opt` (the ZCU104 version does not
fit).

### 2. Find what `fpga_build` packed

```bash
RUN=$(ls -d build/$K/*-bitstream-zedboard | tail -n 1)
echo $RUN
ls $RUN/${K}_opu/
```

This takes the newest ZedBoard build of `$K` by its folder name. Not `build/$K/latest`:
any later run of the kernel, even `fpga_test`, moves `latest` to itself.

The folder must have:

| File | If it is missing |
| --- | --- |
| `$K.bit.bin`, `$K.bit.bin.md5sum` | the build failed: see `$RUN/vivado-console.log` |
| `$K.dtbo` | no `dtc` on the server: install it (above), then `fpga_pynq $K $RUN` packs it again, no rebuild |
| `classify` | no ARM compiler (install it, then `fpga_pynq $K $RUN`), or the kernel has no program in `software/opu/$K/` (see [Another kernel](#another-kernel)) |
| `load.sh`, `$K.dtsi`, `x${K}_hw.h`, `kernel_regs.h` | always there |

### 3. Check that the board is free

The board is shared. See who is on it and what is loaded:

```bash
ssh zb05 'uptime; who; ls /configfs/device-tree/overlays/; cat /sys/class/uio/uio*/name'
```

If someone else is logged in, agree with them first: loading replaces their design.

### 4. Copy the design to the board

```bash
scp $RUN/${K}_opu/* zb05:/media/sd-pl/firmware/
```

It overwrites the previous design's files of the same names; the test image stays.

### 5. Load it

```bash
ssh zb05 sh /media/sd-pl/firmware/load.sh
```

A good load prints:

```text
/media/sd-pl/firmware/K.bit.bin: OK                     the copy arrived intact
removing the overlay /configfs/device-tree/overlays/full the board's own design goes
BIN FILE loaded through FPGA manager successfully
FPGA: operating
FCLK_CLK0 is 99999 kHz: setting it to the IO PLL / 21   the board's clock was 100 MHz
FCLK_CLK0: 76189 kHz, built for 77777 kHz               now at or below the build's clock
kernel AP_CTRL at 0x40000000: 0x00000004 (...)          the kernel answers: idle, ready
```

The clock it prints last must not be above "built for": a kernel run faster than it was
built for can give wrong answers. Then check that Linux made the device `classify` opens:

```bash
ssh zb05 'cat /sys/class/uio/uio*/name'
```

The list must contain `kernel`.

### 6. Run it on the test image

```bash
ssh zb05 "cd /media/sd-pl/firmware && ./classify cube.npy scores.npy reference_fixed.npy"
```

`classify` runs the kernel on the whole cube (in strips of 32-row multiples if it does not
fit in the board's reserved memory, with the same answers), prints the FPGA's time and the
wall-clock time, and compares every pixel with `reference_fixed.npy`: **PASS** or
**FAIL**.

### 7. Keep the results

Into the build's own folder, so they stay with the design they came from:

```bash
scp zb05:/media/sd-pl/firmware/scores.npy $RUN/
```

`scores.npy` is what the kernel wrote: H x W x 4 int32, the 3 scores x 2^16, then the class
(0 cloud, 1 land, 2 sea). Write the timings worth keeping in `docs/experiments/`
(`build/` is not in git).

### 8. Give the board back

```bash
ssh zb05 reboot
```

After the reboot the board runs its own design again.

## Again later, without copying

The files stay on the SD card. After a reboot, load and run in one go:

```bash
ssh zb05 'sh /media/sd-pl/firmware/load.sh && cat /sys/class/uio/uio*/name && cd /media/sd-pl/firmware && ./classify cube.npy scores.npy reference_fixed.npy'
```

## When something goes wrong

| What you see | Why | What to do |
| --- | --- | --- |
| `classify: no UIO device named kernel` | the design is not loaded: the board rebooted (`uptime`), or someone loaded another design | [load again](#again-later-without-copying) |
| `classify: cube.npy: expected '<u2' in C order` | the cube on the board is float32 | the uint16 copy in [Once](#once-before-the-first-test) |
| `md5sum: ... FAILED` | the `.bit.bin` on the board is not the one built | copy again (step 4) |
| `load.sh` stops at `dtc: not found` | the package has no `.dtbo` | step 2's table: install `dtc`, `fpga_pynq $K $RUN`, copy again |
| the clock after `load.sh` is above "built for" | it could not be set | do not run `classify`; paste the output |
| `ssh zb05` asks for a password | the key for `zb05` in `~/.ssh/config` is not used | `grep -A4 -i "^Host zb05" ~/.ssh/config` |
| `ls: cannot access 'build/K/*-bitstream-zedboard'` | no ZedBoard build of `$K` | `fpga_build $K --board zedboard` |

## Another kernel

`load.sh` works for any kernel built with `--board zedboard`. `classify` is
`justounetsimple_opt`'s program (`software/opu/justounetsimple_opt/classify.c`) and works
for a kernel with the same interface: AXI-Lite registers `din0`, `din1`, ... (the raw
cube), `dout` (H x W x 4 int32 scores), `height` and `width`, and at most 4 `m_axi` ports
(the Zynq-7000's HP ports). For a new kernel like that, a folder
`software/opu/NEWKERNEL/` with an `include.txt` holding the line
`software/opu/justounetsimple_opt/classify.c` gives it the same program (see
`software/opu/justounetsimple_opt_zynq7020/include.txt`). A kernel with another interface
needs its own `.c` program in `software/opu/NEWKERNEL/`.

`fpga_ip $K --board zedboard` stops at the IP and makes nothing `load.sh` can load: it is
for putting the kernel into a Vivado design of your own.

## The Zynq-7030 (PicoBoB)

The same steps with the 7030 kernel and its board:

```bash
K=justounetsimple_opt_zynq7030
fpga_build $K --board zynq7030
RUN=$(ls -d build/$K/*-bitstream-zynq7030 | tail -n 1)
```

then steps 3 to 8 against the PicoBoB's address instead of `zb05`.
