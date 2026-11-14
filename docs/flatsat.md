# Testing a design on the HYPSO-2 flatsat

How to run a `_zynq7030` kernel on the flatsat, the ground copy of HYPSO-2: its OPU
(PicoBoB, a Zynq-7030) with the camera, on the CAN bus to `ws2`. The kernel runs on a real
capture (`binned_cube.bip`), and the class of every pixel comes back to the PC. How the
tarball is built and what is in it: [opu.md](opu.md).

Every command below uses `sp_unet_small_opt_u8_zynq7030`, the first kernel run this way
(2026-11-10). For another kernel, change that name everywhere; nothing else changes.

## Three machines, two links

```text
lab server --scp (ethernet)--> ws2 --CAN bus, 1 Mbit/s--> OPU (PicoBoB, Zynq-7030)
                               hypso-cli runs here: the only way to reach the OPU
```

The OPU has no network and no SSH. Everything it does is a command sent from `hypso-cli` over
CAN: files with `opu upload` / `opu download`, Linux commands with `shell remote oneshot`.
On the bus, `ws2` is node 15, the EPS (the power board) node 4, the OPU node 12.

### The remote shell

Every subsystem on the bus (OPU, EPS, UHF, ...) runs a small shell service: it takes a
command line over CAN, runs it, and sends back what it printed (and an exit code).
`hypso-cli` talks to it with three commands (`help shell remote oneshot` etc. in hypso-cli
prints its own help):

| Command | What it does |
| --- | --- |
| `shell remote NODE` | opens a prompt in which every line you type runs on that node; `exit`, `q` or Ctrl-D back to hypso-cli |
| `shell remote oneshot NODE TIMEOUT_S COMMAND ...` | runs one command on that node and returns when its output is in: no prompt to leave, and it is in the log as one line. Every command in this file uses it |
| `shell remote timeout NODE [SECONDS]` | prints the node's own limit on how long a command may run; with `SECONDS`, changes it |

So `shell remote oneshot 12 30 tar -xzf ...` is "on the OPU (node 12), run `tar -xzf ...`,
and wait at most 30 s for its answer". On the EPS (node 4) the commands are the EPS's own,
not Linux: `output 10 1 0` and `hkout` below.

There are **two limits**, and the shorter one wins:

- `TIMEOUT_S` in `oneshot`: how long hypso-cli waits for the answer;
- the node's own limit, from `shell remote timeout`: how long the OPU lets a command run
  before it stops it. **10 s from boot**, until you change it (step 3). It is back to 10 s
  after every boot.

`oneshot 12 300 ...` with the OPU still at 10 s stops the command after 10 s anyway, with
no output.

Three more things the remote shell does:

- **One command at a time.** While one runs, the OPU's shell does not answer, though
  `csp ping 12` does: `shell remote timeout` then fails with `failed: 71 (Protocol error)`.
  Wait for the command to end, at most the OPU's limit.
- **A command can not be stopped.** Ctrl-C quits hypso-cli itself (the OPU, its limit and the
  design loaded stay as they were); the command on the OPU runs on until it ends or reaches
  the limit. A command with nothing to do (`sh` alone, cut off while pasting) waits the whole
  limit: check that the whole line is there before Enter.
- **The command line has a limit** (about 300 characters; hypso-cli says only `command string
  (...) is too long`). Put a long command in a script, upload it once and run the script
  (`MartinOgJacob/cnn-1d-test.sh`, step 6).

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

`fpga_package` needs the build in a dated run (`build/K/picobob/DATE_TIME/`). The
`sp_unet_small_opt_u8_zynq7030` picobob build of 2026-11-11 is not: its files, `export.zip`
among them, are straight in `picobob/`, and `fpga_package` would move `export.zip` into a new,
empty run and stop for want of the firmware. Its package was remade by hand on 2026-11-14,
with the band fix (tarball md5 `782545f1...`), and is on ws2 already: for it, start at step 3.

## 2. Copy it to ws2, from the lab server

Still on the server, in the same folder (`ws2` is in the server's `~/.ssh/config`; it asks for
the `hypso` password):

```sh
scp build/sp_unet_small_opt_u8_zynq7030/picobob/sp_unet_small_opt_u8_zynq7030_opu.tar.gz ws2:MartinOgJacob/
```

It lands in `/home/hypso/MartinOgJacob/` on `ws2`. A plain copy: the OPU knows nothing of it
yet.

## 3. Start hypso-cli and the OPU, on ws2

Everything on `ws2` stays in `/home/hypso/MartinOgJacob/`: the tarball, `hypso-cli`, its
logs (it writes `YYMMDDTHHMMSS_hypso-cli.log` in the folder it is started from) and the
downloaded labels. `hypso-cli` is one standalone program; copy it there once:

```sh
cp /home/hypso/martin/hypso-cli /home/hypso/MartinOgJacob/
```

Then, each time:

```sh
cd /home/hypso/MartinOgJacob
rlwrap ./hypso-cli 15 -ccan0=0/0
```

`15` is ws2's own address on the bus, `-ccan0=0/0` puts it on the `can0` interface, and
`rlwrap` only gives the prompt arrow keys and history.

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
took the command: `shell remote oneshot 4 3 hkout`.

What the three commands do:

- `shell remote oneshot 4 3 output 10 1 0`: on the EPS (node 4), switch on output 10, the
  OPU's power. The OPU then boots Linux and its flight software (opu-services).
- `csp ping 12`: does the OPU answer on the bus yet.
- `shell remote timeout 12 300`: the OPU's own limit from 10 s to 300 s, so the run in step 6
  can finish ("The remote shell", above). It lasts until the OPU is switched off. **Send it
  only once `csp ping 12` answers**, and check that it prints `Previous timeout value: 10`.
  Sent too early it fails (`failed: 71 (Protocol error)`) and the limit stays 10 s: on
  2026-11-13 that is what happened, and the run in step 6 stopped after 13 s with no labels.
  `shell remote timeout 12` (no number) prints the value now in force.

### The live log: the OPU's serial console

Everything the OPU prints, as it prints it: the boot, opu-services, errors from a run. It comes
over a USB cable from the OPU to ws2, not over CAN, so it works before the OPU answers on the
bus and while its remote shell is busy. In a **second terminal on ws2** (a second SSH login, or
a second tmux window: Ctrl-B, then C; Ctrl-B, then a number to switch), before powering on:

```sh
picocom -b 115200 /dev/ttyUSB0
```

- `Terminal ready` and then nothing until the OPU is on: that is right.
- It says the device is busy: someone left a picocom open.
  `sudo killall picocom`, then the command again.
- Quit with **Ctrl-A, then Ctrl-X**. Ctrl-C goes to the OPU, not to picocom.
- Keep it open during the run; close it at the end (step 9), or the next one finds it busy.

The log of a run itself (`cnn_accelerator`'s times) is not here but in a file on the OPU,
read with `cat` (step 6).

## 4. Upload and unpack

```
opu upload /home/hypso/MartinOgJacob/sp_unet_small_opt_u8_zynq7030_opu.tar.gz /media/sd-pl/sp_unet_small_opt_u8_zynq7030_opu.tar.gz
shell remote oneshot 12 30 tar -xzf /media/sd-pl/sp_unet_small_opt_u8_zynq7030_opu.tar.gz -C /media/sd-pl/
shell remote oneshot 12 10 ls -l /media/sd-pl/firmware/ /media/sd-pl/sp_unet_small_opt_u8_zynq7030/
```

- `opu upload FROM TO`: `FROM` on ws2, `TO` on the OPU, sent in pieces over CAN (the
  2.3 MB tarball: a minute or so). Full paths: it does not expand `~`.
- `tar` runs on the OPU and unpacks into `/media/sd-pl/`, the OPU's working folder on its SD
  card: `firmware/K.bit.bin` and `K.dtbo` (the hardware) and `K/` (`cnn_accelerator` and
  `process-capture.sh`, the software).
- `ls` checks that both are there.

`cnn_accelerator` is built with each kernel's own driver: every kernel brings its own, in its
own folder.

## 5. A capture

A capture is a folder with two files: `binned_cube.bip` (the image: lines x samples x
bands, 16 bits a value) and `capture_config.ini` (its size and settings). `cnn_accelerator`
reads the size from the `.ini`, so the folder can be any capture of 120 bands.

`process-capture.sh` takes the folder as a path under `/media/sd-pl/`: `hsi1` is
`/media/sd-pl/hsi1`, `flatsat-dataset/vancouver_2026-09-10T19-16-05Z` is
`/media/sd-pl/flatsat-dataset/vancouver_2026-09-10T19-16-05Z`. It writes the labels and its
log into that folder.

### Finding captures

Every folder on the OPU's SD card with a cube in it:

```
shell remote oneshot 12 60 find /media/sd-pl -maxdepth 3 -name binned_cube.bip
```

`ls -l /media/sd-pl/` alone does not tell: a folder shows as 4096 bytes whatever is in it.
Captures taken here are `hsiN`; they do not stay (on 2026-11-13, `hsi1` of 2026-11-10 was gone).

`/media/sd-pl/flatsat-dataset/` holds real HYPSO-2 captures, named as on the data server
(target, then time). On 2026-11-14, 21 of them:

```text
aquawatchcoral_2026-08-24T03-04-16Z       lakecanyon_2026-08-23T18-48-27Z
aquawatchgrahams_2026-08-23T23-55-00Z     msSpitsbergenmanual_2026-08-24T13-51-46Z
aquawatchmoreton_2026-08-22T23-51-45Z     section7platform_2025-08-17T08-46-06Z
balticAlgae_2026-08-25T10-45-37Z          section7platform_2025-08-26T09-26-31Z
corpuschristi_2026-08-27T17-21-15Z        section7platform_2025-10-31T09-13-58Z
djibouti_2026-08-25T07-45-20Z             tahiti_2026-08-20T20-32-59Z
easter_2026-08-27T17-36-00Z               vancouver_2025-07-17T19-11-01Z
galveston_2026-08-23T17-13-42Z            vancouver_2026-07-30T19-36-20Z
gobabeb_2026-08-20T09-21-52Z              vancouver_2026-09-10T19-16-05Z
hawkebay_2026-08-23T22-20-32Z             vanmijenfjorden_2026-08-27T12-18-48Z
koreanCalValIE_2026-08-25T02-51-03Z
```

To choose: a coast (vancouver, galveston, corpuschristi, hawkebay) has sea and land, often
cloud; gobabeb is desert (land only); tahiti and easter are islands (mostly sea). To see what one
looks like before running, find the same name on the data server and look at its picture.

### Checking one

Before running, that the folder has both files and the cube is 120 bands:

```
shell remote oneshot 12 10 ls -l /media/sd-pl/flatsat-dataset/vancouver_2026-09-10T19-16-05Z
shell remote oneshot 12 10 cat /media/sd-pl/flatsat-dataset/vancouver_2026-09-10T19-16-05Z/capture_config.ini
```

- `column_count = 1080`, `bin_factor = 9`, `sample_divisor = 1`: 1080 / 9 = 120 bands.
- Note the lines and samples: the size of the picture in step 7.
- `binned_cube.bip` is lines x samples x 120 x 2 bytes; smaller, and the capture was cut short.
  `hsi1` on 2026-11-10: 598 x 1092, 156 723 840 bytes.

### Running on a capture from flatsat-dataset

Step 6 and 7 with the folder's path in place of `hsi1`:

```
obip fpga reload sp_unet_small_opt_u8_zynq7030.bit.bin -o sp_unet_small_opt_u8_zynq7030.dtbo
shell remote oneshot 12 10 cat /sys/class/uio/uio*/name
shell remote oneshot 12 300 /media/sd-pl/sp_unet_small_opt_u8_zynq7030/process-capture.sh flatsat-dataset/vancouver_2026-09-10T19-16-05Z 2>&1
obip fpga reset
shell remote oneshot 12 10 tail -n 40 /media/sd-pl/flatsat-dataset/vancouver_2026-09-10T19-16-05Z/logs/sp_unet_small_opt_u8_zynq7030.log
opu download /media/sd-pl/flatsat-dataset/vancouver_2026-09-10T19-16-05Z/sp_unet_small_opt_u8_zynq7030-labels.bin /home/hypso/MartinOgJacob/vancouver-labels.bin
```

For another capture, change `vancouver_2026-09-10T19-16-05Z` everywhere, and the name the
labels get on ws2. `2>&1` brings back the error messages too (the remote shell runs the line
through a shell).

The dataset is shared: once the labels are on ws2, remove what the run added (only those two
files; the folder's own `logs/` stays):

```
shell remote oneshot 12 10 rm /media/sd-pl/flatsat-dataset/vancouver_2026-09-10T19-16-05Z/sp_unet_small_opt_u8_zynq7030-labels.bin /media/sd-pl/flatsat-dataset/vancouver_2026-09-10T19-16-05Z/logs/sp_unet_small_opt_u8_zynq7030.log
```

### Taking a new one

With the OPU's own design loaded (before step 6, or after `obip fpga reset`): the camera's
path into memory is in the OPU's own design, not in yours. 100 frames; `-d b` saves
`binned_cube.bip`, `-k` skips the compression:

```
hsi capture -n 100 -d b -k
shell remote oneshot 12 10 ls -l /media/sd-pl/
```

It goes in the next free `/media/sd-pl/hsiN`: the `ls` shows which. Run on it as above, with
`hsiN` as the path.

## 6. Load, run, restore

With the capture `hsi1`; for one from `flatsat-dataset/`, the same commands with its path
are in step 5, "Running on a capture from flatsat-dataset".

```
obip fpga reload sp_unet_small_opt_u8_zynq7030.bit.bin -o sp_unet_small_opt_u8_zynq7030.dtbo
shell remote oneshot 12 10 cat /sys/class/uio/uio*/name
shell remote oneshot 12 300 /media/sd-pl/sp_unet_small_opt_u8_zynq7030/process-capture.sh hsi1
obip fpga reset
```

- `obip fpga reload`: takes bare names and looks for them in `/media/sd-pl/firmware/` (why the
  tarball has that folder). It loads your bitstream into the FPGA (the PL) while Linux keeps
  running, then applies the device-tree overlay (`.dtbo`), which tells Linux where your
  kernel's registers are.
- `cat .../uio*/name`: the overlay makes those registers a `uio` device, which `cnn_accelerator`
  opens. The names must include `hls_kernel`; otherwise the overlay did not apply: do not run.
- `process-capture.sh hsi1`: runs `cnn_accelerator` on `/media/sd-pl/hsi1`. It reads the size
  from `capture_config.ini`, copies `binned_cube.bip` into memory with each pixel's bands
  reversed (a capture stores them in the reverse order to the `.nc` cubes the networks were
  trained on; see `docs/opu.md`, "The band order"), starts the kernel, waits for it, and writes
  the labels and a log.
- `obip fpga reset` puts the OPU's own design back. **Always run it**, before any capture.

`process-capture.sh` writes `/media/sd-pl/hsi1/sp_unet_small_opt_u8_zynq7030-labels.bin`
(a byte a pixel, HYPSO's values: sea 0, land 1, cloud 2, in the cube's order, as the OPU's
own 1D CNN writes `cnn-1d-labels.bin`) and its log to `/media/sd-pl/hsi1/logs/`:

```
shell remote oneshot 12 10 cat /media/sd-pl/hsi1/logs/sp_unet_small_opt_u8_zynq7030.log
```

What a good log has, in order:

- `Bands reversed in N ms` (`process-capture.sh` logs at debug level, so it is there): the
  reversal described above. A package without it (before 2026-11-14) labels no pixel as sea: see "When something
  goes wrong".
- `Input data loaded in ... ms`: the cube off the SD card, the reversal included.
- `CNN finished in ... ms` and `CNN's own count: ... cycles`.
- `Total runtime ... ms`.

Check the clock once per design, in the log: the kernel's own count of cycles over the ARM's
FPGA time should be the clock it was built for. On 2026-11-14, on balticAlgae (598 x 1092):
loaded in 6 794 ms, the kernel 180.7 ms, 18 066 698 cycles, so 100.0 MHz; 7.30 s in all
(before the band fix). `obip fpga reload` has no fallback when the overlay cannot set the
clock, unlike `load.sh`.

### cnn-1d on the same capture, to compare

The OPU's own 1D CNN (cnn-1d-picobob) is on it already: `firmware/cnn-1d-picobob.bit.bin`
and `.dtbo` (2026-10-17, as `~schimen/cnn-1d-picobob/firmware-to-ws2.tar.gz` on ws2) and
`/media/sd-pl/cnn-1d-accelerator/` (`cnn_accelerator` of 2026-10-23, `cnn_config.ini`,
`rad_matrix_binned_flipped.bin`). Its `process-capture.sh` writes `cnn-1d-labels.bin`,
which the onboard pipeline has already written in the `flatsat-dataset` captures, so run it
through `cnn-1d-test.sh` instead (in `/home/hypso/MartinOgJacob/` on ws2): the same
command, to `cnn-1d-test-labels.bin` and `cnn-1d-test.log`. Upload it once:

```
shell remote oneshot 12 10 mkdir -p /media/sd-pl/MartinOgJacob
opu upload /home/hypso/MartinOgJacob/cnn-1d-test.sh /media/sd-pl/MartinOgJacob/cnn-1d-test.sh
```

Then, as for your kernel:

```
obip fpga reload cnn-1d-picobob.bit.bin -o cnn-1d-picobob.dtbo
shell remote oneshot 12 10 cat /sys/class/uio/uio*/name
shell remote oneshot 12 300 sh /media/sd-pl/MartinOgJacob/cnn-1d-test.sh flatsat-dataset/balticAlgae_2026-08-25T10-45-37Z 2>&1
obip fpga reset
shell remote oneshot 12 10 tail -n 20 /media/sd-pl/flatsat-dataset/balticAlgae_2026-08-25T10-45-37Z/cnn-1d-test.log
```

- The `uio` names must include `cnn_1d_accelerator` and `cnn_1d_memory`.
- In the log: `Input data loaded in ... ms` and `CNN finished in ... ms`. On balticAlgae
  (598 x 1092) on 2026-08-31, cnn-1d's owner had 6 864 ms and 5 905 ms.
- `sam_cnn_picobob.bit.bin` in `firmware/` is an earlier build (2026-06-03): its overlay does
  not name the devices as `cnn_accelerator` looks for them, so it does not run with it.

## 7. The labels back to the PC

In hypso-cli:

```
opu download /media/sd-pl/hsi1/sp_unet_small_opt_u8_zynq7030-labels.bin /home/hypso/MartinOgJacob/labels.bin
```

The reverse of `opu upload`: `FROM` on the OPU, `TO` on ws2, over CAN.

Then on the laptop (598 1092: the capture's height and width, from step 5):

```sh
scp ws2:/home/hypso/MartinOgJacob/labels.bin .
echo '{"lines": 598, "samples": 1092}' > labels-meta.json
python software/opu/scripts/show_result.py labels.bin labels.png labels-meta.json
start labels.png
```

`show_result.py` (cnn-1d-picobob's) colours sea blue, land green, cloud grey. A fresh capture has
no `-meta.json`, so `labels-meta.json` gives it the size. For a capture from the data server, its
own `-meta.json` and `--scaled` with its scaled-radiance picture lay the labels as that picture
(`docs/opu.md`, "The labels as a picture").

### Checking the band fix, on balticAlgae

The labels the bit-exact model gives on balticAlgae's cube with its bands reversed are on the lab
server: `~/Documents/MartinOgJacob/flatsat-results/2026-11-14/balticAlgae-model-bands_reversed-labels.bin`
(24.8 % sea, 94.4 % the same as cnn-1d). The board's must be the same, byte for byte. Run on
`flatsat-dataset/balticAlgae_2026-08-25T10-45-37Z` (step 5, "Running on a capture from
flatsat-dataset"), with the labels downloaded as `/home/hypso/MartinOgJacob/balticAlgae-labels.bin`.
Then on the lab server:

```sh
cd ~/Documents/MartinOgJacob/flatsat-results/2026-11-14
scp ws2:MartinOgJacob/balticAlgae-labels.bin balticAlgae-bandfix-labels.bin
cmp balticAlgae-bandfix-labels.bin balticAlgae-model-bands_reversed-labels.bin && echo identical
```

`identical`: the fix works on the board. `differ: byte ...`: send the log.

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

## 9. Clean up, for the next one

The flatsat is shared: leave the OPU as you found it. Before step 8, once the labels are
on ws2, in hypso-cli (each line under the remote shell's length limit):

```
obip fpga reset
shell remote oneshot 12 10 cd /media/sd-pl/flatsat-dataset/balticAlgae_2026-08-25T10-45-37Z && rm cnn-1d-test* sp_unet* logs/sp_unet*
shell remote oneshot 12 10 rm /media/sd-pl/sp_unet_small_opt_u8_zynq7030_opu.tar.gz
shell remote oneshot 12 60 find /media/sd-pl/flatsat-dataset -maxdepth 3 -name "*sp_unet*" -o -name "cnn-1d-test*"
```

- The second line for each capture you ran on: only your files; the pipeline's
  `cnn-1d-labels.bin` and the folder's own `logs/` stay. The last prints nothing when
  they are all gone.
- **The designs stay**, ready for the next run: `firmware/K.bit.bin`, `.md5sum`, `.dtbo`,
  `/media/sd-pl/K/` and `/media/sd-pl/MartinOgJacob/cnn-1d-test.sh` (cnn-1d's own,
  `firmware/cnn-1d-picobob.*` and `/media/sd-pl/cnn-1d-accelerator/`, are not yours: never
  remove them). `/media/sd-pl/bitstreams/` is others' too.
- An `hsiN` you took yourself: `rm -r` it too, if nobody else needs it.

Then `opu shutdown` (step 8): it also puts the OPU's limit back to 10 s. On ws2: quit
hypso-cli and picocom (Ctrl-A, Ctrl-X: an open picocom keeps `/dev/ttyUSB0` busy for the
next one), and `rm /home/hypso/MartinOgJacob/*.download-format`.

Over SSH, run hypso-cli in `tmux new -s hypso`: a dropped connection then leaves it running
(`tmux attach -t hypso`), mid-download too. A dropped `opu download` continues from its
`.download-format` file when sent again with the same destination.

## When something goes wrong

| What you see | Why | What to do |
| --- | --- | --- |
| `csp ping 12` fails | the OPU is off or still booting | step 3: power it on, wait, ping again |
| a remote command stops after 10 s, or `process-capture.sh` returns at once with no output and no labels | the OPU's own limit is still 10 s (`shell remote timeout` was not sent, or failed before the OPU answered) | `shell remote timeout 12`, then `shell remote timeout 12 300` until it prints the previous value; run again |
| the run returns with no labels; the log says `Configuration file not found: /media/sd-pl/hsiN/capture_config.ini` | no capture in `hsiN` (gone since, or never there); the run itself made `hsiN/logs/` | find one (`shell remote oneshot 12 60 find /media -name binned_cube.bip`) or take one (step 5), and run on that name |
| `shell remote timeout` failed: 71 (Protocol error) | sent before the OPU answered; or, if `csp ping 12` answers, the OPU's shell is still running an earlier command | `csp ping 12` until it answers, then again; if it answers, wait for the earlier command's limit (at most 300 s) and again; still failing: power cycle (step 8) |
| `command string (...) is too long` | the command line is over the remote shell's limit | put it in a script (step 6, `cnn-1d-test.sh`) |
| back at the `ws2` prompt after Ctrl-C | Ctrl-C quits hypso-cli; the OPU is as it was, a command on it runs on | start hypso-cli again; wait for the command's limit before the next |
| `K initialize failed: Device not found` | the design or its overlay is not loaded | `obip fpga reload K.bit.bin -o K.dtbo`, then the `uio` check |
| the cube's size does not match | the cube is not what `capture_config.ini` says (not 120 bands, or cut short) | read `capture_config.ini` and `ls -l` the capture (step 5) |
| the labels have no sea, or look nothing like the capture | the bands in the wrong order: a package from before 2026-11-14, whose `cnn_accelerator` does not reverse them (the old run on balticAlgae: 0 % sea) | package again (step 1) and upload it (steps 2 and 4); check with "Checking the band fix" (step 7) |
| the kernel does not finish | the design is not running (clock, wrong bitstream) | check the clock (step 6); `obip fpga reset` |

More, for the build and cnn_accelerator: [opu.md](opu.md), "When something goes wrong".
