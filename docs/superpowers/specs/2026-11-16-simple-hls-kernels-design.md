# Simple HLS kernels for the ZCU104 (design)

2026-11-16. Status: for review.

## Why

Another paper compares an HLS accelerator against the Vitis AI DPU on the ZCU104. The
optimized kernels (`*_opt`, hand-built streaming modules in `src/optimized/`) stay the
thesis's own work. For the paper we want a second, plainer HLS version of each network: code
that reads like the golden kernels, made fast with the standard pragmas only. It must beat
the DPU and be slower than our best design.

DPU times to beat (ms per 598 × 1092 image, the user's table 5.5.1) next to our measured best
(`optimizations_implemented/speed_zcu104.md`, 250 MHz):

| Model | DPU | `_opt` (int8, no packing) | best |
|---|---|---|---|
| 1D-SpectralPointwise | 168.4 | 12.1 | 12.1 |
| 2D-Justo-UNet-Simple | 188.8 | 19.3 (`_tiled`) | 19.3 |
| SpecPW-UNet small | 233.6 | 26.2 | 19.3 (`_u8`) |
| SpecPW-UNet medium | 267.9 | 93.1 | 46.6 (`_u8`) |
| SpecPW-UNet large | 291.5 | 186.1 | 93.1 (`_u8`) |

## Goals, non-goals, anti-goals

Goals:
- One `_simple` kernel each for `spectralpointwise1d`, `justounetsimple`, `sp_unet_small`,
  `sp_unet_medium`, `sp_unet_large`, for the ZCU104 (board `zcu104`).
- int8 quantized weights, integer arithmetic, **bit for bit the same answers as the model's
  `_opt` kernel** (`classify_fixed`), so the existing testbench checks, `classify.ipynb` and
  `tools/models.py`'s `fixed_scores` serve them unchanged.
- Faster than the DPU on every model; slower than our best design. Large at ~200 ms is fine
  (the user: "if it runs at 200 ms that's fine").
- The code reads like the golden kernels: one function per layer on whole arrays, plain loops.

Non-goals:
- JustoLiuNet and WaveletCNN (no DPU number), the Zynq-7000 boards, uint8 activations.
- Matching `_opt`'s speed or resources.

Anti-goals (would fail the point even if it works):
- Using the `src/optimized/` modules (line buffers, `pixel_buffer`, `conv3x3_tiled`, ...).
- Two multiplies per DSP (`dsp_pack.hpp`) or any other hand-packed arithmetic.
- Many templates or hand-built streaming chains: the point is "golden + pragmas".

## Constraints

- Pragmas: PIPELINE, DATAFLOW, UNROLL, ARRAY_PARTITION, plus what the interface and storage
  need (INTERFACE, STREAM/BIND_STORAGE where an array's depth or memory type must be set).
- The kernel interface is the one `classify.ipynb` drives: `din0` (it finds any number of
  `dinN` ports in the driver header), `dout`, `height`, `width`, `cycles`; one start a whole image.
- Raw cube in (uint16, 120 bands, `cube_word_t` words as `_opt`), the z-score in hardware with
  `src/optimized/zscore_int8.hpp`'s A/B table (generated data, not a module), so the time
  measured is the whole job, as for `_opt`.
- ZCU104: 1 728 DSPs, 312 BRAM36 (11 Mb), 96 URAM (27 Mb). Timing at 250 MHz (4 ns), as `_opt`.

## Approaches considered

- **A. One function per layer on whole-patch arrays, DATAFLOW across the layers (chosen).**
  Each layer a process; the arrays between them become ping-pong buffers, so every layer
  works on a different patch at once. Time a patch = the slowest layer. Golden-shaped code.
  Costs on-chip memory (below).
- B. The layers one after another, DATAFLOW only for read → network → write. Simplest, least
  memory, but the time a patch is the sum of the layers: large would be 3-5× slower than A
  and lose to the DPU. Rejected.
- C. Pixel streams with line buffers: what `_opt` already is. Rejected (anti-goal).

The user picked A.

## Design

### Files

```text
src/simple/layers.hpp                         the few templates (below), with pragmas
src/models/1D-SpectralPointwise/spectralpointwise1d_simple/spectralpointwise1d_simple.{cpp,hpp}
src/models/2D-Justo-UNet-Simple/justounetsimple_simple/justounetsimple_simple.{cpp,hpp}
src/models/SpecPW-UNet/{Small,Medium,Large}/sp_unet_SIZE_simple/sp_unet_SIZE_simple.{cpp,hpp}
tb/K_simple/K_simple_tb.cpp                   run_kernel() + the _opt kernel's check header
optimizations_implemented/simple.md           the five kernels: factors, predicted T, then measured
```

`scripts/` find them by name (`find_kernel`), `tools/models.py` maps `K_simple` to its model by
prefix and treats it as optimized (`fixed_scores`, `classify.ipynb`): no changes there expected.

### The templates (`src/simple/layers.hpp`)

1. **`conv<H, W, IN, OUT, K, UI, UO, ACT, shifts..., weights, bias>(in, out)`**: 1×1 or 3×3,
   same padding; the integer arithmetic of `specpw_unet_fixed_golden.hpp`'s
   `conv2d_same_fixed_golden` (bias shifted up, exact sum, round half up, ReLU and saturate, or
   none for the scores), with the fractions as template constants as in `_opt`'s `layer_fracs`.
   Loops: pixel → output chunk → tap → input chunk, the whole nest flattened and
   **PIPELINE II=1**; inside, **UNROLL** UI inputs × UO outputs (UI·UO multipliers). The taps
   are *not* unrolled: each cycle reads UI channels of one pixel, so the input array is only
   **ARRAY_PARTITION**ed on its channel dimension (cyclic UI) and there is no window to build;
   the output cyclic UO. Border taps read zero (an `if`, as the golden). Cycles a patch:
   `H·W·K²·⌈IN/UI⌉·⌈OUT/UO⌉`. The weights are constant arrays, partitioned to match.
   An optional second input for the concat (below).
2. **`maxpool2<H, W, C>(in, out)`**: 2×2, stride 2, pipelined.
3. **`read_zscore<...>(din, patch, ...)`**: a patch's rows from DDR (each patch row one burst
   of consecutive addresses), bands 8-117 kept, `(dn·A + B) >> FRAC`, saturated: the int8
   z-scores. Its edge handling (repeat the edge where the last patch sticks out) must equal
   `_opt`'s `read_patch_rows_simple`.
4. **`write_scores<...>(scores, dout, ...)`**: the patch's scores and class to DDR, the part
   outside the image dropped, the same words as `_opt`'s `store_scores`.

Upsample is no template: the conv after it reads `in[y/2][x/2]` (a template flag), which saves
an array. The concat likewise: `double4_conv1` takes `up2` and the skip as two inputs, and an
input chunk comes wholly from one (UI divides DOUBLE3), so the choice is which statement runs,
not a run-time bank (the project's HLS lesson).

### SpectralPointwise

The same `conv` with K = 1 on a block of 1 024 consecutive pixels (H = 1, W = 1 024): a block
is contiguous in DDR, so the read is one long burst, no patches. The last block is partial
(the count `height·width`); the writer drops the rest.

### The SpecPW-UNet skip

The skip (double1's output) is read by the pool now and by `double4_conv1` ~10 processes
later. With patch-level DATAFLOW it must hold about as many patches as there are layers
between: a copy process sends one array to the pool and one to a deep buffer
(`#pragma HLS STREAM ... type=pipo depth=N`, N the number of processes between, in URAM).
Large: ~10 × 64 KB.

### Sizing: UI and UO

One per-patch budget T for each kernel; every layer gets the smallest power-of-two-ish UI×UO
(dividing its channels) with cycles ≤ T. Budgets (my call, to be checked against csynth):

| Kernel | T (cycles a patch / block) | predicted at 250 MHz | `_opt` |
|---|---|---|---|
| spectralpointwise1d_simple | read-bound, one port | ~40-50 ms | 12.1 |
| justounetsimple_simple | ~20 000 | ~55 ms | 19.3 |
| sp_unet_small_simple | ~20 000 | ~55 ms | 26.2 |
| sp_unet_medium_simple | ~45 000 | ~120 ms | 93.1 |
| sp_unet_large_simple | ~75 000 | ~200 ms | 186.1 |

Large at 75 000 needs ~1 100 multipliers (its ~82 M multiplies a patch ÷ T), about what `_opt`
uses; the other four take far fewer DSPs than `_opt`.

### Memory: the main risk

Every array between processes is a ping-pong pair. Large: ~430 channels at 32×32 and ~350 at
16×16, int16 activations, int8 z-scores: ~8.5 Mb, ×2 = ~17 Mb, plus the skip ~5 Mb: ~22 of
the 38 Mb (BRAM + URAM). Fitting needs the big arrays in URAM (`BIND_STORAGE ... impl=uram`)
and not partitioning them more than UI/UO need (each bank at least one BRAM18). Upsample and
concat fused into the convs (above) save ~3 Mb. Small and medium are well inside. If large does
not fit, the fallback is to merge adjacent 1×1 stem layers into one process (fewer buffers), not
to go back to streaming.

### DDR

One read port (`din0`) for all five: simplest, and a patch-row burst is 7 680 bytes. Expected
3-5 GB/s, ~35-50 ms for the 157 MB cube: under every budget but SpectralPointwise's, which is
then read-bound (still well under the DPU).

### Tests

- `scripts/test.sh K_simple` for each: `tb/K_simple/` calls the model's `_opt` check header
  (`justounetsimple_opt_check.hpp`, `sp_unet_opt_check.hpp`, `spectralpointwise1d_opt_check.hpp`)
  through a `run_kernel` with `_opt`'s signature (din0 for every port it passes): bit for bit
  the integer reference and close to the float golden one.
- Mutation-test once: a wrong shift, a dropped border tap, the concat order swapped must fail.
- `python -m unittest discover -s tests`, adding `_simple` where tests list kernels.
- The user's runs: csynth (Final II per process, DSP/BRAM/URAM), cosim on one (arithmetic
  changed from `_opt`), then the board with `classify.ipynb`.

## What is left to the builder

- The exact UI/UO per layer from the budgets, written as a table in each `.cpp` (as `_opt`'s).
- The skip buffer depth, from the process count.
- Whether HLS flattens the conv nest by itself or it is written as one loop with counters.
- Whether `read_zscore` and the first conv need their own z-score array or share one.

## Calls made without the user (check these)

- The name `_simple`; the folder `src/simple/`.
- Bit for bit `_opt`'s answers (int16 activations in SpecPW-UNet, as `_opt`), not `_u8`'s.
- Raw cube in and the z-score in hardware, one read port.
- Taps not unrolled (no 3×3 window), so partitioning is on channels only.
- Upsample and concat fused into the convs' reads; maxpool its own process.
- The budgets T in the table above.
