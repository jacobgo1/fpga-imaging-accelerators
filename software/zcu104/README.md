# ZCU104 board scripts

Headless bring-up, A53 benchmarking, and running the PL kernel, all driven
from the command line -- no Etcher, no serial-terminal GUI, no manual
copying.

```text
flash-sdcard.sh          write an image (Ubuntu or PYNQ) to the SD card (dd/bmaptool)
make-boot-seed.sh        preload cloud-init for headless SSH -- Ubuntu image only, see below
deploy-and-run.sh        rsync this repo to a remote host and run a main.py stage there
export_weights.py        dev machine: PyTorch checkpoint -> the .npz the board-side scripts expect
run_justoliunet.py       on the board (PYNQ, has network/OS): load a bitstream, feed an image, read the result
program-jtag.sh          program just the PL bitstream over JTAG (no OS needed on the board)
pack_data_bin.py         dev machine: weights.npz + an input tile -> one raw binary blob for JTAG download
jtag_run.tcl             xsct: push that blob into DDR and run the kernel with no OS on the board at all
unpack_result.py         decode jtag_run.tcl's dout.bin into numbers
```

Sections 1-4 below are the **A53 software baseline** (plain Ubuntu image).
Section 5 is **running the actual FPGA kernel with network/ssh access to
the board's own Linux** (PYNQ). Section 5b is the same, but for a board
reachable **only over JTAG** -- no OS, no network, a different and more
experimental recipe. Pick whichever matches your actual setup.

## 1. Get an image

Download AMD/Canonical's prebuilt Ubuntu Server image for ZCU104 from
`ubuntu.com/download/amd-xilinx` (pick the arm64 ZCU104 build; a 22.04+
release gives you Python 3.10+, which this repo's `main.py` needs). Nothing
here downloads it for you -- versions and filenames change, the scripts just
take the file you already have.

## 2. Flash it

```bash
./flash-sdcard.sh ubuntu-*.img.xz /dev/diskN            # dry run: prints the plan
./flash-sdcard.sh ubuntu-*.img.xz /dev/diskN --yes       # actually writes it
```

Find the right device first with `diskutil list` (macOS) or `lsblk` (Linux).
The script refuses partitions (trailing number) and, on Linux, refuses a
device with your own root filesystem mounted on it -- but the device
argument itself is on you to get right.

## 3. Seed it for headless SSH access

Before first boot, with the card's boot partition mounted:

```bash
./make-boot-seed.sh /Volumes/system-boot ~/.ssh/id_ed25519.pub
# or with a static IP instead of DHCP:
./make-boot-seed.sh /Volumes/system-boot ~/.ssh/id_ed25519.pub zcu104 192.168.1.50/24 192.168.1.1
```

This only works on Canonical's official image (it ships cloud-init's
NoCloud datasource reading `user-data`/`network-config` from the boot
partition -- the same mechanism Ubuntu uses on Raspberry Pi). Eject the
card, boot the board, and within a minute or two:

```bash
ssh ubuntu@zcu104.local     # mDNS; or the static/DHCP IP directly
```

No monitor, keyboard or UART cable needed. If mDNS doesn't resolve on your
network, check your router's DHCP lease list instead.

## 4. Build and run on the A53

```bash
./deploy-and-run.sh ubuntu@zcu104.local
```

Rsyncs the repo (minus `build/`, `reports/`, `.git`) to
`~/fpga-imaging-accelerators` on the board, runs
`python3 main.py native --kernel justoliunet` there with the board's own
native `g++` (no cross-compiler -- the board *is* the target
architecture), and rsyncs `reports/` back so the run's `manifest.json` and
console logs land locally next to every other run's, per the main
[README](../../README.md#where-the-reports-are).

```bash
# any main.py stage/kernel works the same way:
./deploy-and-run.sh ubuntu@zcu104.local -- native --kernel justoliunet
# tune the benchmark rep count (see tb/justoliunet/justoliunet_tb.cpp):
JNET_BENCH_REPS=2000 ./deploy-and-run.sh ubuntu@zcu104.local
```

The board's first run needs `build-essential` and `git`, which
`make-boot-seed.sh`'s cloud-init config already installs; if you skipped
that step, `ssh ubuntu@<host> sudo apt install -y build-essential git`
once before the first `deploy-and-run.sh`.

## 5. Run the FPGA kernel (PL) -- PYNQ

Build the bitstream on a machine with Vitis HLS + Vivado installed first
(one command does csim -> csynth -> cosim -> export -> synth -> impl ->
bitstream, using [`boards/zcu104/system.tcl`](../../boards/zcu104/system.tcl)):

```bash
python3 main.py bitstream --kernel justoliunet --config config/project.local.json
```

Flash a **PYNQ** image (not the Ubuntu one) with the same `flash-sdcard.sh`
from step 2 -- PYNQ boots with SSH already enabled (default `xilinx`/
`xilinx`, reachable at `xilinx@pynq.local`), so skip `make-boot-seed.sh`.

Copy the build's outputs and get your trained model onto the board:

```bash
scp artifacts/justoliunet/<run>/deliverables/bitstream/system.bit  xilinx@pynq.local:~/
unzip -p artifacts/justoliunet/<run>/deliverables/hardware/system.xsa '*.hwh' > /tmp/system.hwh
scp /tmp/system.hwh xilinx@pynq.local:~/

python3 export_weights.py your_model.pt weights.npz    # dev machine, needs torch
scp weights.npz xilinx@pynq.local:~/
scp your_tile.npy xilinx@pynq.local:~/                 # optional: a real hyperspectral tile
scp run_justoliunet.py xilinx@pynq.local:~/
```

Then, on the board:

```bash
ssh xilinx@pynq.local
python3 run_justoliunet.py system.bit weights.npz --input your_tile.npy
python3 run_justoliunet.py system.bit weights.npz              # no --input: fixed random test tile
```

`run_justoliunet.py` needs only `pynq` + `numpy` (already on a PYNQ image);
`export_weights.py` needs `torch` and only ever runs on your dev machine.
If register names come back wrong (`find_register` in `run_justoliunet.py`
lists what the IP actually has), that's `boards/zcu104/system.tcl` or the
HLS-generated names differing from what this script guesses -- both are
easy one-line fixes once you can see the real names.

## 5b. Build server with a JTAG cable straight to the board, no OS on it

Different situation, different recipe: no SD card, no Linux, no network to
the board at all -- the server's Vivado/Vitis talks to it purely over JTAG.
That means no filesystem or Python on the board either, so weights and the
input image have to go in as raw memory writes, and DDR itself isn't even
usable until something runs `psu_init` (normally the FSBL's job on an
SD-card boot; there is no FSBL here).

```text
program-jtag.sh      program just the PL bitstream over JTAG (Vivado Hardware Manager)
pack_data_bin.py      dev machine: weights.npz + an input tile -> one raw binary blob
jtag_run.tcl          xsct: psu_init, push the blob into DDR, poke registers, read dout back
unpack_result.py       decode jtag_run.tcl's dout.bin into numbers
```

```bash
# 1. Program the PL (server, Vivado's settings64.sh sourced):
./program-jtag.sh artifacts/justoliunet/<run>/deliverables/bitstream/system.bit

# 2. Pack weights + image into one blob (dev machine, needs the weights.npz
#    from export_weights.py):
python3 pack_data_bin.py weights.npz --input your_tile.npy -o data.bin

# 3. Fill in the register/address TODOs (see the file itself for exactly
#    where each value comes from -- most need your real csynth/export
#    output, they cannot be guessed):
cp justoliunet_regs.tcl.example justoliunet_regs.tcl
$EDITOR justoliunet_regs.tcl

# 4. Run it (server, Vitis' settings64.sh sourced, so `xsct` is on PATH):
xsct jtag_run.tcl data.bin path/to/psu_init.tcl justoliunet_regs.tcl
python3 unpack_result.py dout.bin
```

**Be aware this path is a first draft, more so than anything else here.**
`jtag_run.tcl` was written without a real board or Vitis install to check
`xsct`'s exact command syntax against -- the DDR byte offsets inside it are
verified (they come straight out of `pack_data_bin.py`'s own math, tested
above), but `kernel_base` and every register offset in
`justoliunet_regs.tcl` are placeholders you fill in from your real build,
and commands like `dow -data`/`mrd -bin`/`psu_init` may need adjusting
against `xsct`'s own `help <command>` for your installed version. Send me
the actual error and we'll fix it -- that's the expected next step here,
not a sign something was done wrong.
