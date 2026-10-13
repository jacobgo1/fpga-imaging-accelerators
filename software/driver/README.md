# jopt_driver: justounetsimple_opt from Linux on the board

What the `justounetsimple_opt` notebook does, as one C++ program for the board's
ARM, without Python or PYNQ: the same program for the ZCU104 (64-bit ARM,
kernel `justounetsimple_opt`) and HYPSO-2 (Zynq-7030, 32-bit ARM, kernel
`justounetsimple_opt_zynq7030`). Modelled on `driver_ntnu/cnn-1d-picobob`, the
1D CNN driver flown on HYPSO-2:

1. the raw cube (`.npy` from `tools/justounetsimple_image.py`, or the BIP file
   the HYPSO payload writes) goes from the file into reserved DDR;
2. the kernel gets the addresses and the size through its registers, starts,
   and the driver polls the done bit;
3. the scores and classes go to files, and with `--reference` they are checked
   bit for bit against the PC's integer model.

If the reserved DDR cannot hold the whole cube, it runs in strips of a multiple
of 32 rows (whole patches, so the answer is the same), in two buffers: the ARM
copies strip s + 1 while the FPGA does strip s.

```text
src/main.cpp          options, the strips, timing, files, the check
src/accelerator.*     the kernel's control registers: addresses, start, done
src/formats.hpp       the cube and score formats both kernels share
src/registers.*       the register offsets, read from the kernel's x*_hw.h at run time
src/device.*          physical memory: UIO, /dev/mem, or plain memory (sim)
src/image_files.*     .npy, raw BIP, HYPSO capture_config.ini
src/sim.*             jopt_driver_sim: a kernel's C++ instead of the FPGA
firmware/*.dtsi       device tree entries for the UIO devices, per board
scripts/              server.sh (all of the server's part), cross-compile.sh, package.sh,
                      process-capture.sh (HYPSO-2), Dockerfile
jopt_driver.ipynb     the ZCU104 (PYNQ) notebook packed with the driver: Run All
```

The register offsets come from the `x<kernel>_hw.h` of the bitstream's own HLS
build (`xjustounetsimple_opt_hw.h`, `xjustounetsimple_opt_zynq7030_hw.h`), like
in the notebook; by default the one next to the program. So a rebuilt kernel
needs no new driver, and neither kernel needs its own: the header says how many
`din` ports there are (4 on the ZCU104, 2 on the 7030), and the driver points
them all at the cube. The driver refuses
a header with a kernel argument it doesn't know, because that argument would
silently stay 0.

## The short way

On the lab server, after `git pull`, one script does all of the server's part
(the weights if missing, the tests, both bitstreams, the cross compilers, both
drivers, both packages). Bitstreams take hours, so run it with `nohup`, which
survives a dropped connection; `--skip-bitstreams` packages the newest existing
ones instead:

```bash
nohup software/driver/scripts/server.sh > /dev/null 2>&1 &
tail -f software/driver/server.log
```

On the ZCU104, `jopt_driver_zcu104.tar.gz` holds `jopt_driver.ipynb`: upload the
archive next to `aegean_unet`, unpack it (`!tar -xzf jopt_driver_zcu104.tar.gz`),
open `jopt_driver_zcu104/jopt_driver.ipynb` and Run All. The sections below are
the same steps one at a time.

## Build

On the lab server (Linux), from the repository root:

```bash
software/driver/scripts/cross-compile.sh zcu104
software/driver/scripts/cross-compile.sh zynq7030
```

This builds `software/driver/build-zcu104/jopt_driver` (aarch64) and
`build-zynq7030/jopt_driver` (armhf), linked statically so that they run on
PYNQ's Ubuntu and on HYPSO-2's Petalinux alike. The script uses the cross
compiler on PATH (`sudo apt install g++-aarch64-linux-gnu g++-arm-linux-gnueabihf`),
or Docker if there is none.

Then pack the driver with the newest bitstream run of that board's kernel (its
register header and `.bit`, plus the device tree snippet). The runs come from
`fpga_build justounetsimple_opt` and `fpga_build justounetsimple_opt_zynq7030 --board zynq7030`:

```bash
software/driver/scripts/package.sh zcu104       # -> software/driver/build-zcu104/jopt_driver_zcu104.tar.gz
software/driver/scripts/package.sh zynq7030     # -> software/driver/build-zynq7030/jopt_driver_zynq7030.tar.gz
```

## Test on the PC

`python3 -m unittest tests.test_driver` builds `jopt_driver_sim` (the driver with
a kernel's C++ compiled in place of the FPGA), once per kernel, and checks it
against `classify_fixed`. It covers one start, strips, raw BIP with
`capture_config.ini`, a wrong answer, fewer ports, and unknown arguments. It
tests everything except the hardware: UIO, `/dev/mem` and the real DDR. By hand:

```bash
cmake -S software/driver -B software/driver/build-sim -DJOPT_SIM=ON -DJOPT_SIM_KERNEL=justounetsimple_opt
cmake --build software/driver/build-sim
software/driver/build-sim/jopt_driver_sim --header build/justounetsimple_opt/latest/justounetsimple_opt_pynq/xjustounetsimple_opt_hw.h --input aegean_unet/cube.npy --reference aegean_unet/reference_fixed.npy
```

The header comes from a bitstream build, so this needs one (`fpga_build justounetsimple_opt`).

## ZCU104 with PYNQ

PYNQ's device tree has no UIO entries for the kernel, so the driver uses
`/dev/mem`, with the kernel's registers at `0xA000_0000` (where
`boards/zcu104/system.tcl` puts them). For DDR it borrows a buffer that PYNQ
allocates: physically contiguous, and Linux leaves it alone for as long as the
notebook holds it. Upload `jopt_driver_zcu104.tar.gz` next to the image folder
in Jupyter, then in cells:

```python
!tar -xzf jopt_driver_zcu104.tar.gz
import numpy as np
from pynq import Overlay, allocate
overlay = Overlay('jopt_driver_zcu104/justounetsimple_opt.bit')   # programs the FPGA and sets pl_clk0
ddr = allocate(shape=(128 * 1024 * 1024,), dtype=np.uint8)        # 128 MB for the driver, kept while it runs
!jopt_driver_zcu104/jopt_driver --devmem-memory {ddr.physical_address}:{ddr.nbytes} --input aegean_unet/cube.npy --reference aegean_unet/reference_fixed.npy --labels aegean_unet/driver_labels.npy
```

Load the bitstream with PYNQ's `Overlay`, not `fpgautil`: the `Overlay` also
sets `pl_clk0` to the clock the kernel was built for. If `/dev/mem` refuses the
buffer ("Operation not permitted": a kernel built with `STRICT_DEVMEM`), reserve
DDR at boot instead: `mem=1536M` on the kernel command line leaves
`0x6000_0000`–`0x7FFF_FFFF` to the driver (`--devmem-memory 0x60000000:0x20000000`).

## ZCU104 with Petalinux

Add `firmware/zcu104.dtsi`: part 1 goes in the base device tree
(`system-user.dtsi`), part 2 with the bitstream. Add
`uio_pdrv_genirq.of_id=generic-uio` to the bootargs. The driver then finds both
UIO devices by name, and needs no addresses:

```bash
./jopt_driver --input cube.npy --labels labels.npy
```

## HYPSO-2 (Zynq-7030)

As with the 1D CNN: `firmware/zynq7030.dtsi` becomes the `.dtbo` next to the
bitstream (opu-system's `prepare_pl_firmware.sh`). It gives the kernel's
registers and HYPSO-2's reserved CubeDMA memory (152 MB at `0x2000_0000`, the
region `driver_ntnu` uses) to UIO. A 598 x 1092 capture does not fit in that
whole, so it runs in 3 strips of 256 rows.

```sh
# on the OPU, after the archive is unpacked in /media/sd-pl and the bitstream loaded
/media/sd-pl/jopt_driver_zynq7030/process-capture.sh hsi0
```

That classifies `/media/sd-pl/hsi0/binned_cube.bip` (its size from
`capture_config.ini`) into `hsi0/jopt-labels.bin`, in the HYPSO tools' label
order (0 sea, 1 land, 2 cloud; `driver_ntnu/.../show_result.py` reads it), with
the log in `hsi0/logs/jopt-driver.log`.

What is not there yet: `justounetsimple_opt_zynq7030` still has to be built into
HYPSO-2's flight block design, as `driver_ntnu`'s `firmware/cnn-1d-picobob.tcl`
does for the 1D CNN; `boards/zynq7030/system.tcl` makes a design of its own, not
HYPSO-2's. If the flight design names the block something other than `kernel`,
change the `&kernel` label in `firmware/zynq7030.dtsi`.

## Options

`jopt_driver --help` lists them. Without UIO: `--devmem-memory ADDRESS:SIZE`, and
`--board zynq7030` for registers at `0x4000_0000` (or `--devmem-regs ADDRESS`).
Output: `--labels` (uint8, 0 cloud, 1 land, 2 sea; `--hypso-labels` for HYPSO's
order), `--scores` (the kernel's H x W x 4 int32), each `.npy` or raw by its name.
Exit codes: 0 done, 1 error, 2 different from `--reference`.

## Timing

The driver prints the time for each part. Reading the cube from the SD card
and copying it into DDR takes far longer than the FPGA. The copy is also slow
because the DDR is mapped uncached: CPU and FPGA see the same bytes without
cache flushes, but each write goes to DDR one by one. If that dominates, the
fix is a cached mapping with explicit cache maintenance, or having the payload
capture into the reserved DDR directly. It isn't the kernel.
