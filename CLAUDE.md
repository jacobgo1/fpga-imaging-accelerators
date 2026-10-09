# CLAUDE.md

## What this project is

Neural networks for HYPSO hyperspectral images (cloud / land / sea per pixel),
turned into FPGA accelerators with Vitis HLS. Two targets:

- **ZCU104** (Zynq UltraScale+ ZU7EV), the default board: the family HYPSO-3
  will fly, and what runs in the lab (PYNQ).
- **Zynq-7030** (`--board zynq7030`): HYPSO-2's FPGA. Much smaller (400 DSPs, no
  UltraRAM, 64-bit HP ports), so kernels must be sized for it. See
  `boards/README.md`.

Trained models are the `.pt` files in `weights/fp32/`. **Never delete them**: all
weights are extracted from them on demand.

## Where things run

- **Laptop (Windows, Git Bash):** editing, `python main.py native` (g++) and the
  Python tests. No AMD tools.
- **Lab server (Linux):** `source fpga.sh`, then `fpga_build K` for csynth and
  bitstreams. The user runs these and pastes results; Claude cannot.
  - HLS log: `build/K/latest/hls.log`
  - reports: `build/K/latest/hls/solution/syn/report/`
- **Board:** the ZCU104 with PYNQ. Upload `K_pynq.zip`, unzip, run the notebook.

## Layout

```text
main.py              the build: python3 main.py STAGE --kernel K [--board B]
fpga.sh              short commands for the server (Tab completes kernels, runs, boards)
src/hls/K/           one folder per kernel; tb/K/ its self-checking testbench
src/golden/          plain-loop reference layers and models (what is computed)
src/optimized/       streaming, integer fixed-point building blocks (how it is done fast)
src/common/          hls::stream shim so kernels compile with plain g++
boards/B/system.tcl  one block design per board, shared by every kernel
software/pynq/K/     the notebook for kernel K
tools/               weights, numpy reference models, image preparation (tools/README.md)
tests/               Python tests of all of the above, no AMD tools needed
docs/framework.md    what each build step does
```

Kernels now: `justounetsimple` (golden baseline), `justounetsimple_opt` (fast),
`matmul` (minimal example, used by the framework tests).

## How a model is built (follow justounetsimple_opt)

1. Quantize the weights: `python tools/quantize_weights.py weights/fp32/MODEL`
   (int8, BatchNorm folded, power-of-two scales).
2. Write a bit-exact numpy model of the integer kernel in `tools/` (see
   `classify_fixed` in `tools/justounetsimple_model.py`). It is the reference
   the board is checked against.
3. Write the kernel as a DATAFLOW chain from `src/optimized/` blocks:
   - read the raw cube from DDR and do the preprocessing in hardware;
   - integer fixed point throughout;
   - size each stage so none waits for another.
4. Testbench: bit-exact against a plain-loop integer reference, and close to the
   float golden model. Mutation-test it: inject bugs, check it fails.
5. The notebook: one start for the whole image; print wall-clock and FPGA time.

## Before saying something works

Claude runs only the local checks below, never csynth, cosim or bitstream:
those are the user's, on the lab server, every time.

- `python -m unittest discover -s tests`: every test must pass.
- `python main.py native --kernel K` for each changed kernel.
- Say what still needs the user's build. Timing and resources only come from
  their runs: ask for the relevant `hls.log` lines (`Final II`, burst inference,
  warnings) instead of guessing.
- Measure the whole path, not only the kernel: once the board's ARM is in the
  loop, it dominated wall-clock time (10 s of Python vs 0.05 s of FPGA).

## HLS lessons from this project

- An array index that selects a memory bank must be a constant in the code.
  Choosing the bank at run time duplicates the stores, e.g. II = 5 in the
  patch buffer. Put run-time choices in the address within a bank, or in which
  statement runs.
- `#pragma HLS DEPENDENCE ... false` when reads and writes provably never meet
  (ping-pong halves, line buffers).
- Bursts need consecutive addresses in one loop: read whole patch rows.
- Vitis warns that UltraRAM does not use read-first mode; BRAM is the fallback.
- Two int8 multiplies fit in one DSP when they share an operand
  (`src/optimized/conv3x3_packed.hpp`, AMD WP486): it needs activations in
  -127..127 and the UltraScale+ DSP48E2, so not on the Zynq-7030. Check the DSP
  count in the csynth report: about 864 for conv1 means the packing mapped.
  Pack the weights once at load, not next to the multiply: HLS put that addition
  in the DSP pre-adder and the RTL was slightly wrong (csim passed, cosim failed).
- csim passing does not prove the RTL. After changing arithmetic, ask the user
  for a cosim (`python3 main.py cosim --kernel K`); a mismatch there is
  reproducible in minutes, a mismatch on the board is not.
- HLS knows the weights' values (loaded once from constant arrays) and sizes
  each multiplier for its weight: small weights become LUT shift-adds, zero
  weights vanish. That is why DSP counts can be far below the multiply count.
- Generated files must stay in sync:
  - `src/hls/justounetsimple_opt/justounetsimple_prep.hpp` comes from
    `python tools/justounetsimple_model.py`, and a test checks it;
  - `weights/quantized/` comes from `quantize_weights.py` and is not in git.

## Working with the user

- Do not commit. The user reviews every change first.
- Shell commands the user will paste must be complete as written: no `<placeholders>`.
- Keep the code comments' style: short, explaining why, plain English.
