# ZCU104 board scripts

Headless bring-up, A53 benchmarking, and running the PL kernel, all driven
from the command line -- no Etcher, no serial-terminal GUI, no manual
copying. Five scripts:

```text
flash-sdcard.sh     write an image (Ubuntu or PYNQ) to the SD card (dd/bmaptool)
make-boot-seed.sh   preload cloud-init for headless SSH -- Ubuntu image only, see below
deploy-and-run.sh   rsync this repo to the board and run a main.py stage there (A53 track)
export_weights.py   dev machine: PyTorch checkpoint -> the .npz run_justoliunet.py expects
run_justoliunet.py  on the board (PYNQ): load a bitstream, feed an image, read the result
```

Sections 1-4 below are the **A53 software baseline** (plain Ubuntu image).
Section 5 is **running the actual FPGA kernel**, which needs a PYNQ image
instead -- skip straight there if that's what you're after.

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
