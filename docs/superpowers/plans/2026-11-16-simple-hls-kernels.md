# Simple HLS kernels (ZCU104) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Five `_simple` kernels (spectralpointwise1d, justounetsimple, sp_unet small/medium/large) that give `_opt`'s answers bit for bit, written as golden-style per-layer functions with only PIPELINE / DATAFLOW / UNROLL / ARRAY_PARTITION.

**Architecture:** One function per layer on whole-patch arrays, DATAFLOW across the layers (ping-pong buffers). A conv pipelines pixel → output chunk → tap → input chunk at II=1 and unrolls UI inputs × UO outputs; arrays are partitioned on channels only. Four templates in `src/simple/layers.hpp`.

**Tech Stack:** C++14 / Vitis HLS 2025.2, g++ testbenches (`scripts/test.sh`), Python unittest.

**Spec:** `docs/superpowers/specs/2026-11-16-simple-hls-kernels-design.md`

## Global Constraints

- Answers bit for bit those of the model's `_opt` kernel; the existing check headers prove it.
- No `src/optimized/` modules; only its generated data headers (`zscore_int8.hpp`, and for the 1D
  kernel `justoliunet_opt_formats.hpp`, `spectralpointwise1d_opt_formats.hpp`) and the shims in `src/common/`.
- No DSP packing, no `mul<DSP>`/`BIND_OP`: HLS chooses the multipliers.
- Pragmas: PIPELINE, DATAFLOW, UNROLL, ARRAY_PARTITION, INTERFACE, STREAM (pipo depth),
  BIND_STORAGE (URAM for the big arrays). Nothing else.
- Interface: `void K_simple(const cube_word_t* din0, score_word_t* dout, int height, int width)`;
  m_axi `din0` on `gmem0`, `dout` on `gmem1`, everything on s_axilite `control`; `depth=` for cosim
  as `_opt` (30000 / 2000 for the 2D, the 1D's from its `_opt`). No `cycles` register: the
  notebook then times with the host timer (it handles a kernel without one).
- `cube_word_t` = `hls::vector<uint16_t, 8>`, `score_word_t` = `hls::vector<int32_t, 4>` (3 scores, class).
- Per-kernel budget T (cycles a patch, each layer ≤ T): justounetsimple 20 000, sp_unet_small 20 000,
  medium 45 000, large 75 000; spectralpointwise1d: UI·UO so it outruns its read.
- Comment style: short, why, plain English. Do not commit (the user reviews first): the
  "commit" steps below are "stop and list the changed files".

## Review Focus

1. Image not a multiple of 32 (598 × 1092): the last patch row/column must repeat the edge on read
   and drop the outside on write exactly as `_opt` — the check headers' 40 × 50 image covers it; keep it.
2. A 3×3 conv's border taps (patch edge, not image edge) read zero, for every UI chunk — the
   checks cover it; the mutation test in Task 1 must show a dropped border check fails.
3. Concat order (up2's channels first, then the skip): swapped gives 30-40 % accuracy, not a crash —
   Task 2's mutation step.
4. SpectralPointwise's last block shorter than the block (63 and 17 pixels in its check) — its check covers it.
5. `IN` not divisible by UI (110 bands): the last chunk's extra lanes must contribute zero — pick UI
   dividing 110 (2, 5, 10, 11, 22, 55, 110) and `static_assert` it in `conv`.

---

### Task 1: `src/simple/layers.hpp` and `justounetsimple_simple`

**Files:**
- Create: `src/simple/layers.hpp`
- Create: `src/models/2D-Justo-UNet-Simple/justounetsimple_simple/justounetsimple_simple.{hpp,cpp}`
- Create: `tb/justounetsimple_simple/justounetsimple_simple_tb.cpp`

**Interfaces (produces, in namespace `simple`):**
- `template<int H, int W, int IN, int OUT, int K, int UI, int UO, bool RELU, int ACT_MAX, int BIAS_SHIFT, int SHIFT, bool UPSAMPLED, const int8_t (&WEIGHT)[OUT][IN][K][K], const int8_t (&BIAS)[OUT], typename TI, typename TO> void conv(const TI in[UPSAMPLED ? H/2 : H][UPSAMPLED ? W/2 : W][IN], TO out[H][W][OUT])`
  — same padding; sum (int64 in C, `ap_int` wide enough is fine) starts at `BIAS << BIAS_SHIFT`;
  output = round half up by `SHIFT` (`(s + (1 << (SHIFT-1))) >> SHIFT`), then if RELU clamp to
  `[0, ACT_MAX]`; without RELU the rounded value (the scores, `TO` = int32). `UPSAMPLED` reads
  `in[y/2][x/2]`. `static_assert(IN % UI == 0 && OUT % UO == 0)`. Loop nest flattened, PIPELINE II=1,
  inner UI×UO UNROLLed; `in` partitioned cyclic UI on dim 3, `out` cyclic UO on dim 3, weights
  copied once into a local array partitioned to match. Cycles: `H·W·K²·(IN/UI)·(OUT/UO)`.
- `conv2` — the same with two inputs `a[H][W][IA]`, `b[H][W][IB]`, input channels `a` then `b`
  (the concat); `UI` divides `IA` and `IB`, so each chunk's source is fixed by an `if`.
- `template<int H, int W, int C, typename T> void maxpool2(const T in[H][W][C], T out[H/2][W/2][C])`, PIPELINE II=1 over output pixels, C unrolled.
- `template<int P, int BANDS, int WPP, int FIRST_WORD, ...zscore constants...> void read_zscore(const cube_word_t* din, int pr, int pc, int height, int width, int8_t patch[P][P][BANDS])`
  — patch (pr, pc)'s rows, one burst per row (`read_patch_rows_simple`'s addressing and edge
  repetition), bands 8-117, `(dn·A[k] + B[k]) >> FRAC` saturated to ±IN_MAX (int8 path:
  `zscore_int8`). Write `patch` cyclic-partitioned to the first conv's UI.
- `template<int P, int CLASSES> void write_scores(const int32_t s[P][P][CLASSES], score_word_t* dout, int pr, int pc, int height, int width)`
  — the pixels inside the image, row by row, scores then the first highest class (`store_scores`'s words, without its rounding: the conv already rounded).
- The top: `for (patch) { DATAFLOW region: read_zscore → conv… → write_scores }` — put the per-patch
  chain in its own function with `#pragma HLS DATAFLOW` and call it in a loop over patches, or the
  whole loop as the dataflow region with the patch index passed down; whichever HLS overlaps
  across patches (check csynth's dataflow viewer / interval). Record the choice in a comment.

- [ ] **Step 1: Testbench.** `tb/justounetsimple_simple/justounetsimple_simple_tb.cpp`: include
  `justounetsimple_simple.hpp`, define `run_kernel(cube_word_t*, score_word_t*, int h, int w)` calling
  `justounetsimple_simple(cube, scores, h, w)`, include `../justounetsimple_opt/justounetsimple_opt_check.hpp`.
  The .hpp must define the names that header uses: `PATCH_SIZE, HALF_PATCH, CUBE_BANDS, MODEL_BANDS, BASE_CHANNELS, CLASSES, BANDS_PER_WORD, WORDS_PER_PIXEL, ZSCORE_FRAC_BITS=5, ACTIVATION_FRAC_BITS=11, SCORE_FRAC_BITS=16`, the two word types.
- [ ] **Step 2:** `bash scripts/test.sh justounetsimple_simple` — expected: compile error (no kernel).
- [ ] **Step 3: Implement** `layers.hpp` and the kernel: conv1 3×3 110→6 ReLU, maxpool, conv2 6→12
  ReLU, maxpool, conv3 (UPSAMPLED) 12→6 ReLU, conv4 (UPSAMPLED) 6→3 scores; the
  fractions from the weights as `_opt`'s .cpp does. UI/UO table in the .cpp, each ≤ 20 000 cycles,
  e.g. conv1 UI=55 UO=6 (18 432).
- [ ] **Step 4:** `bash scripts/test.sh justounetsimple_simple` — expected `PASSED: justounetsimple_simple`.
- [ ] **Step 5: Mutation test** (revert each after): drop the border `if` in `conv`; shift by `SHIFT-1`;
  read the edge pixel wrong in `read_zscore`. Each must FAIL the testbench.
- [ ] **Step 6:** stop; list changed files.

### Task 2: `sp_unet_small_simple`

**Files:** Create `src/models/SpecPW-UNet/Small/sp_unet_small_simple/sp_unet_small_simple.{hpp,cpp}`,
`tb/sp_unet_small_simple/sp_unet_small_simple_tb.cpp`.

**Interfaces:** consumes Task 1's `conv`, `conv2`, `maxpool2`, `read_zscore`, `write_scores`.
The .hpp defines what `sp_unet_opt_check.hpp` uses: `PATCH_SIZE, CUBE_BANDS, MODEL_BANDS, CLASSES, BANDS_PER_WORD, ZSCORE_FRAC_BITS=5, ACTIVATION_FRAC_BITS=10, SCORE_FRAC_BITS=16`, word types, `model_size` comes from the golden .cpp the tb includes.

- [ ] **Step 1:** tb as `tb/sp_unet_small_opt/sp_unet_small_opt_tb.cpp` (include the kernel's .hpp, then
  `sp_unet_small_golden/sp_unet_small.cpp`, `run_kernel`, `run_golden`, `../sp_unet_small_opt/sp_unet_opt_check.hpp`, `check_kernel()`).
- [ ] **Step 2:** `bash scripts/test.sh sp_unet_small_simple` — expected compile failure.
- [ ] **Step 3: Implement** the chain of `specpw_unet_fixed_golden` (tb/sp_unet_small_opt/): convsp1-3 (K=1),
  double1 ×2, a copy process writing the skip and the pool's input, maxpool, double2 ×2, maxpool,
  up1 (UPSAMPLED), double3 ×2, up2 (UPSAMPLED), double4_conv1 = `conv2(up2, skip)`, double4_conv2,
  final (K=1, no ReLU). act_max 32767, int16 activations. Skip: `#pragma HLS STREAM variable=skip type=pipo depth=N`
  with N = processes between the copy and double4_conv1 + 1, `BIND_STORAGE ... impl=uram`. UI/UO table ≤ 20 000.
- [ ] **Step 4:** `bash scripts/test.sh sp_unet_small_simple` — expected PASSED.
- [ ] **Step 5: Mutation:** swap the concat order in `conv2`; must FAIL.
- [ ] **Step 6:** stop; list changed files.

### Task 3: `sp_unet_medium_simple`, `sp_unet_large_simple`

**Files:** Create both kernels' folders and `tb/` folders as Task 2 (`Medium/`, `Large/`).

**Interfaces:** the small kernel's chain. Put the chain in `src/models/SpecPW-UNet/Small/sp_unet_small_simple/sp_unet_simple.hpp` as one template on the sizes and a UI/UO table type, so each size's .cpp is its table and the top function (as `_opt` shares `sp_unet_opt.hpp`). Do this refactor in this task, re-running small's test.

- [ ] **Step 1:** both testbenches (copy of small's with the size changed).
- [ ] **Step 2:** `bash scripts/test.sh sp_unet_medium_simple` and `..._large_simple` — expected compile failure.
- [ ] **Step 3:** tables: medium ≤ 45 000, large ≤ 75 000 cycles a layer; big arrays (32×32 with ≥ 16 channels) `BIND_STORAGE impl=uram`.
  Write each table's cycles and multipliers as comments, and the sum of multipliers (large ≈ 1 100).
- [ ] **Step 4:** all three `scripts/test.sh sp_unet_{small,medium,large}_simple` PASSED.
- [ ] **Step 5:** stop; list changed files.

### Task 4: `spectralpointwise1d_simple`

**Files:** Create `src/models/1D-SpectralPointwise/spectralpointwise1d_simple/spectralpointwise1d_simple.{hpp,cpp}`, `tb/spectralpointwise1d_simple/spectralpointwise1d_simple_tb.cpp`.

**Interfaces:** `conv` with H=1, W=1024 (block of pixels), K=1. Its own read (`read_zscore_1d`, in
`layers.hpp`): block b's pixels are consecutive in DDR, one burst; the 12-bit z-score of
`justoliunet_opt_formats` (IN_FRAC 8, IN_MAX 2047, PREP_FRAC 19); and `write_scores_1d` writing only
the pixels < `height·width`. Layers: convsp1 110→24, convsp2 24→12 (uint8, ACT_FRAC1/2 = 4, ACT_MAX 255),
convsp3 12→3 scores at OUT_FRAC 16, as `spectralpointwise1d_opt_check.hpp`'s reference.
The .hpp defines `CUBE_BANDS, BANDS_PER_WORD, WORDS_PER_PIXEL`, word types.

- [ ] **Step 1:** tb: `run_kernel(cube_word_t* din, score_word_t* dout, int h, int w, uint32_t& cycles)` sets
  `cycles = 0` and calls the kernel; include `../spectralpointwise1d_opt/spectralpointwise1d_opt_check.hpp`; `main` returns `check_kernel()`.
- [ ] **Step 2:** `bash scripts/test.sh spectralpointwise1d_simple` — expected compile failure.
- [ ] **Step 3:** implement; UI/UO e.g. convsp1 UI=22 UO=24 → 5 cycles a pixel... choose so ≤ 4 cycles a pixel (the read).
- [ ] **Step 4:** PASSED. **Step 5:** mutation: write the last block's padding pixels to DDR → must FAIL (or show the check does not catch it, and say so).
- [ ] **Step 6:** stop; list changed files.

### Task 5: Docs and the full check

**Files:** Create `optimizations_implemented/simple.md` (layout of the other model files: per kernel the
UI/UO table, predicted T and ms at 250 MHz, DSPs designed, "Changes" with 2026-11-16). Modify
`CLAUDE.md` (kernel list: the five `_simple` kernels, one paragraph) and `optimizations_implemented/README.md`
if it lists the model files.

- [ ] **Step 1:** write the docs.
- [ ] **Step 2:** `python -m unittest discover -s tests` — every test passes (fix any test that lists kernels).
- [ ] **Step 3:** all five `bash scripts/test.sh K_simple` PASSED.
- [ ] **Step 4:** report: the user's next commands, complete as written:
  `fpga_build justounetsimple_simple zcu104` (and the others), the `hls.log` lines wanted
  (`Final II` per process, the dataflow interval, DSP/BRAM/URAM), and `fpga_cosim justounetsimple_simple zcu104` for one.
