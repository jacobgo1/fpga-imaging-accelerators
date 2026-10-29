# justoliunet_opt: design, performance model and what to try next

Notes on the optimized 1D-Justo-LiuNet kernels: `justoliunet_opt` and `justoliunet_bn_opt`
for the ZCU104 (committed in `5ae5ec0`, 2026-10-16), and `justoliunet_opt_zynq7020` and
`justoliunet_bn_opt_zynq7020` for the lab's ZedBoards (2026-10-19, `--board zedboard`).
Why they are built the way they are, a performance model, predictions for the Zynq-7030
(HYPSO-2), and the ideas that use the 1D (spectral) model better than the 2D design does.

**Status:** all four kernels C++ bit-exact (testbench and the 10 275 labelled aegean
pixels) and giving the same answers on both chips; not yet synthesized. Everything about speed and resources below is a prediction, except where
it says "cycle model" (a cycle-by-cycle simulation of the DATAFLOW chain, ideal DDR).
Replace predictions with csynth, cosim and board numbers as they come in, and record
measured runs in `docs/experiments/`.

## 1. The workload

Per pixel; $P_l$ = output positions the network uses (the pools drop an odd last
position, so conv1 uses 102 of its 105 outputs, conv2 46 of 47, and bands 107-109 never
reach the output), $W_l = OC_l \cdot IC_l \cdot K$ weights, $M_l = P_l W_l$ MACs, $K = 6$.

| Layer | $P_l$ | $IC_l \to OC_l$ | $W_l$ | $M_l$ | Share |
|---|---|---|---|---|---|
| conv1 | 102 | 1 → 6 | 36 | 3 672 | 6.4 % |
| conv2 | 46 | 6 → 12 | 432 | 19 872 | 34.6 % |
| conv3 | 18 | 12 → 18 | 1 296 | 23 328 | 40.7 % |
| conv4 | 4 | 18 → 24 | 2 592 | 10 368 | 18.1 % |
| dense | 1 | 48 → 3 | 144 | 144 | 0.3 % |
| **Total** | | | 4 500 | $M = 57\,384$ | |

That is ~9× the 2D `justounetsimple` (~6 600 MACs/pixel): the 2D kernel is bound by
memory, this one by arithmetic. A full capture is $N = 598 \cdot 1092 = 653\,016$ pixels.

## 2. The ZCU104 design as built

- **Image partitioning:** none needed. A 1D model looks only at a pixel's own spectrum,
  so the image is one run of $H \cdot W$ pixels in memory order: no patches, halos,
  padding or reorder buffer.
- **Memory:** two read ports (HP0, HP1) take alternate blocks of 16 pixels (240 words,
  one AXI burst each), requested 2 blocks ahead (`read_pixels_ahead`, the
  `read_rows_ahead` idea), so DDR latency is paid once per start. Scores go out through
  HP2. Same formats as `justounetsimple_opt` (raw uint16 cube in, `[H][W][4]` int32 out).
- **Compute:** a DATAFLOW chain, one pixel per stage. Each layer gets just enough
  multipliers to finish a pixel in ≤ 18 cycles. Inputs sit in shift registers (every
  index a constant, no multiplexers), and pooling is a separate small process, so no
  loop-carried dependency limits II.
- **DSPs where the work is:** conv2 and conv3 (75 % of the MACs) on DSPs, two
  multiplies per DSP48E2 (two output channels share a uint8 activation). conv1 (12-bit
  input, not packable), conv4, dense in LUTs. ~1 304 of 1 728 DSPs. Every layer's
  parallelism and DSP/LUT choice is a constant in `justoliunet_opt.hpp` (`CONV4_PACKED`
  moves conv4 to 324 more DSPs).
- **Formats** (chosen on the labelled aegean pixels, `tools/justoliunet_model.py`):
  12-bit z-score input with 8 fraction bits (the 2D kernel's int8/5 bits cost
  justoliunet_bn 1.2 points of agreement with fp32); uint8 activations after each block,
  fraction bits per model (justoliunet 4, 4, 2, 3; justoliunet_bn 4, 4, 2, 2); int32 sums;
  scores × 2¹⁶. Fixed point vs fp32: justoliunet 99.66 % same class (accuracy 96.73 % vs
  96.61 %), justoliunet_bn 99.07 % (96.18 % vs 96.67 %).

| | conv1 | conv2 | conv3 | conv4 | dense | z-score |
|---|---|---|---|---|---|---|
| $\mathit{POS} \cdot \mathit{OCP} \cdot \mathit{FP}$ | 6·6·6 | 3·12·36 | 1·18·72 | 1·6·108 | 3·3 | 8 lanes |
| $m_l$ (multipliers) | 216 | 1 296 | 1 296 | 648 | 9 | 8 |
| $S_l$ (cycles/pixel) | 17 | 16 | **18** | 16 | 16 | 15 |
| Implementation | LUT | 648 DSP | 648 DSP | LUT | LUT | 8 DSP |
| $u_l$ (busy) | 94 % | 85 % | 100 % | 89 % | 89 % | – |

Cycle model: exactly 18.00 cycles/pixel in steady state (conv3 stalled 0.13), ~86
cycles to fill; 18.7 with 200 cycles of DDR latency before every burst.
**$t = 653\,016 \cdot 18 / 200\,\mathrm{MHz} = 58.8$ ms.**

What to read from the first builds: cosim PASS and latency (~63 · 18 + ~100 cycles for
the testbench's 9 × 7 image); every `step` loop at II = 1 in `hls.log`; ~648 DSPs each
for conv2 and conv3 (more means a packed multiply did not map to one DSP); LUTs and
timing; the notebook's "cycles per pixel" on the board (target 18.0).

## 3. Performance model

Per layer, three parallelism factors: $\mathit{POS}$ output positions, $\mathit{OCP}$
output channels and $\mathit{FP}$ of the $IC \cdot K$ taps at once.

$$S_l = \left\lceil \frac{P_l}{\mathit{POS}_l}\right\rceil \cdot \frac{OC_l}{\mathit{OCP}_l} \cdot \frac{IC_l K}{\mathit{FP}_l}, \qquad m_l = \mathit{POS}_l \cdot \mathit{OCP}_l \cdot \mathit{FP}_l$$

Pixel interval, time, throughput, latency:

$$T = \max\big(S_\mathrm{zscore}, S_1, \dots, S_4, S_\mathrm{dense}, T_\mathrm{mem}\big), \qquad
t = \frac{N T + T_\mathrm{fill}}{f_\mathrm{clk}}, \qquad \text{throughput} = \frac{f_\mathrm{clk}}{T}$$

$T_\mathrm{fill} \approx \sum_l S_l + \text{pipeline depths} + L_\mathrm{DDR}$ (a few hundred
cycles; also one pixel's latency).

DSPs ($\rho_l$ = multiplies per DSP, 2 packed, 1 not):
$$D = \sum_l m_l / \rho_l + D_\mathrm{zscore} \le D_\mathrm{chip}$$

Packing two multiplies in one DSP: with shift $S$ (18 on DSP48E2, 16 on DSP48E1), a
group of $G$ products can be summed before they are split:
$$G_{\max} = \left\lfloor \frac{2^{S-1}-1}{a_{\max}\, w_{\max}} \right\rfloor$$

| DSP | uint8 activations ($a \le 255$) | int8 ($a \le 127$) |
|---|---|---|
| DSP48E2 (ZCU104), $S = 18$ | 4 | 8 |
| DSP48E1 (7020/7030), $S = 16$ | 1 | 2 |

Utilization and efficiency:
$$u_l = \frac{M_l}{T\, m_l}, \qquad \eta = \frac{M}{T \sum_l m_l}$$

Bounds: compute and memory ($B = 240$ bytes read per pixel, $n_p$ ports of $w$ bytes per
cycle, 16 on the ZCU104, 8 on the Zynq-7000, efficiency $e \approx 0.8$):
$$T \ge \frac{M}{\rho D + m_\mathrm{LUT}}, \qquad T_\mathrm{mem} = \frac{B}{n_p\, w\, e}, \qquad
m_\mathrm{LUT} \approx \frac{0.7\, L_\mathrm{chip} - L_\mathrm{infra}}{c}$$

with $c \approx 50$ LUTs for a general 8×8 LUT multiplier and $L_\mathrm{infra} \approx$ 15-30 k
LUTs (estimates).

## 4. The Zynq-7000 boards

### The ZedBoard (Zynq-7020) design as built

`justoliunet_opt_zynq7020`, `justoliunet_bn_opt_zynq7020` (`--board zedboard`; 78.1 MHz,
220 DSP48E1, 53.2 k LUTs, 64-bit HP ports). The same techniques as the ZCU104 (every layer
about the same cycles a pixel, DSPs on conv2 and conv3, two multiplies per DSP, uint8,
read-ahead bursts) and the same formats, so the same answers bit for bit. What had to change:

- **Folding.** Each layer takes one output position at a time and spreads it over steps:
  $\mathit{OCP}$ output channels and $\mathit{ICP}$ input channels a cycle, all 6 taps, the
  input-channel chunks summed in an accumulator (`conv1d_stream.hpp`).
- **Spectral streaming.** From conv1 on, a layer's input arrives one position at a time
  and the layer keeps a window of the last 6 (a 1D line buffer, as `conv3x3_stream` keeps
  two image lines). Whole feature maps in wide FIFOs and registers, as on the ZCU104,
  would have cost ~15 k of the 7020's 53 k LUTs (estimate); positions cost ~2 k.
- **Packing on the DSP48E1** (25 × 18 bits): shift 16, so with uint8 every product is
  split on its own ($G = 1$, section 3).
- **One read port** (30 words of 4 bands a pixel, blocks of 8 pixels a burst); conv4,
  conv1, dense and the z-score in LUTs.

| | conv1 | conv2 | conv3 | conv4 | dense | z-score |
|---|---|---|---|---|---|---|
| Per cycle | 1 pos · 6 out · 6 taps | 6 out · 6 in · 6 | 6 out · 6 in · 6 | 2 out · 9 in · 6 | 1 · 3 | 4 bands |
| $m_l$ | 36 | 216 | 216 | 108 | 3 | 4 |
| $S_l$ (cycles/pixel) | 102 | 96 | **108** | 96 | 48 | 30 |
| Implementation | LUT | 108 DSP | 108 DSP | LUT | LUT | LUT |

$S_l = (K-1) + P_l \cdot \frac{OC}{\mathit{OCP}} \cdot \frac{IC}{\mathit{ICP}}$ for the streamed
layers, of which the last output's G·C − 1 steps that read nothing take the next pixel's
first positions (up to 5), so only conv2 still fills 4 (2026-10-19; before, 113). 216 of
220 DSPs. Cycle model: 108 cycles/pixel
in steady state (conv3), unchanged with 300 cycles of DDR latency per burst; with the
position FIFOs at the default depth of 2 instead of 8 it would be 134 (the windows are
filled 5 positions back to back). **$t = 653\,016 \cdot 108 / 78.1\,\mathrm{MHz} \approx 0.90$ s**
(the paper design below, conv4 on DSPs and nothing streamed, gave 1.35 s).

Debugging note: Apple clang 15 at `-O2` (its SLP vectorizer) miscompiled a group choice
written as `for gg: if (gg == g) out[gg * OCP + j] = ...` (the testbench failed; `-O1`,
`-O3` and `-O2 -fno-slp-vectorize` passed). Written as `out[g * OCP + j]`, the same hardware
in HLS, it is right at every level; both `conv1d_stream` and `pool_assemble` use that form.

### The Zynq-7030 design as built

`justoliunet_opt_zynq7030`, `justoliunet_bn_opt_zynq7030` (`--board zynq7030`; 100 MHz):
the ZedBoard design, wider. A streamed layer takes in at most one position a cycle, so
conv1's 102 outputs were the floor; here conv1 is a whole-map `conv1d_engine` at 2
positions a cycle (`pool_adjacent` pools each pair as it comes).

| | conv1 | conv2 | conv3 | conv4 | dense | z-score |
|---|---|---|---|---|---|---|
| Per cycle | 2 pos · 6 out · 6 taps | 12 out · 6 in · 6 | 18 out · 3 in · 6 | 4 out · 6 in · 6 | 1 · 3 | 4 bands |
| $m_l$ | 72 | 432 | 324 | 144 | 3 | 4 |
| $S_l$ (cycles/pixel) | 51 | 51 | **74** | 72 | 48 | 30 |
| Implementation | LUT | 216 DSP | 162 DSP | LUT | LUT | LUT |

378 of 400 DSPs. Cycle model: 74 cycles/pixel (93 with FIFO depth 2), +300 cycles once
with 300 cycles of DDR latency per burst. **$t = 653\,016 \cdot 74 / 100\,\mathrm{MHz} \approx 0.48$ s**
(the paper design below: 81, 0.53 s). conv2 has no step count between 51 (all channels a
step) and 96 (half), so its DSPs are busy 62 %.

### Predictions (paper designs, folding over channels and taps, DSPs only)

These keep whole feature maps and weights from small ROMs. **Section 5 A revises the
folding direction**: folding over positions instead allows hard-wired weights.

**Zynq-7030** (HYPSO-2, 100 MHz, 400 DSPs, 78.6 k LUTs):

| | conv1 | conv2 | conv3 | conv4 | dense | z-score |
|---|---|---|---|---|---|---|
| $\mathit{POS} \cdot \mathit{OCP} \cdot \mathit{FP}$ | 2·6·6 | 2·4·36 | 2·2·72 | 1·4·36 | 2 | 4 lanes |
| $m_l$ | 72 | 288 | 288 | 144 | 2 | 4 |
| $S_l$ | 51 | 69 | **81** | 72 | 72 | 30 |
| Implementation | LUT | 144 DSP | 144 DSP | 72 DSP | LUT | 4 DSP |
| $u_l$ | 63 % | 85 % | 100 % | 89 % | 89 % | – |

$T = 81$, $t \approx 0.53$ s, 364 of 400 DSPs, $\eta = 89$ %, $T_\mathrm{mem} = 37.5$ with one
64-bit port. The ZedBoard design above suggests the 7030 version should also stream
positions and put conv4 in LUTs.

**Zynq-7020, the first paper design** (superseded by the design as built, kept for
comparison; 78.1 MHz, 220 DSPs):

| | conv1 | conv2 | conv3 | conv4 | dense |
|---|---|---|---|---|---|
| $\mathit{POS} \cdot \mathit{OCP} \cdot \mathit{FP}$ | 1·6·6 | 1·4·36 | 1·2·72 | 1·2·36 | 1 |
| $m_l$ | 36 | 144 | 144 | 72 | 1 |
| $S_l$ | 102 | 138 | **162** | 144 | 144 |
| Implementation | LUT | 72 DSP | 72 DSP | 36 DSP | LUT |

$T = 162$, $t \approx 1.35$ s.

**Summary** (full capture; the DSP-only figures are the designs as built (cycle model), the rest are estimates; "with LUT mult." fills the LUT budget with
general multipliers):

| Board | $f_\mathrm{clk}$ | $T$, DSP only | Time | $T$, with LUT mult. | Time |
|---|---|---|---|---|---|
| ZCU104 | 200 MHz | 18 | **58.8 ms** | ~10 (memory-bound) | ~31 ms |
| Zynq-7030 | 100 MHz | 74 | **0.48 s** | ~36 | ~0.24 s |
| Zynq-7020 (ZedBoard) | 78.1 MHz | 108 | **0.90 s** | ~66 | ~0.55 s |

Further differences on the Zynq-7000: 64-bit HP ports, so 30 words of 4 bands a pixel
(as in the 2D 7020/7030 kernels; the z-score then takes 30 cycles, still far below $T$);
no UltraRAM (not needed: 4.5 KB of weights, small maps); DSP48E1 packing has $G = 1$
with uint8 (each product split in LUTs) or $G = 2$ with int8 activations (measured cost
about 0.3-0.5 points of agreement with fp32, with the old 8-bit input); $t \propto T/f$,
so a faster clock than 78 MHz on the 7020 is worth trying.

## 5. Using the 1D model better than the 2D design

What differs from the 2D model: each pixel stands alone; the convolutions run along the
spectrum; conv1 has 1 input channel and the work sits in conv2/conv3; the whole model is
only 4 500 weights; the cube's capture order (each pixel's bands contiguous) is already
the order the model needs. Already used: no patches/halos/reorder, pixel pipelining,
receptive-field pruning, DSPs by work share, uint8 after ReLU so every layer packs.

### A. Hard-wire the weights (largest gain)

A multiplier that only ever sees one weight is a few shifts and adds. The 2D conv1 could
not use this (its multipliers cycled through 4 chunks of bands); the 1D model is small
enough to give every weight its own hard-wired multiplier. Cost with the real weights,
(nonzero canonical-signed-digit digits − 1) adders per weight:

| | conv1 | conv2 | conv3 | conv4 | dense |
|---|---|---|---|---|---|
| justoliunet: adders/weight | 1.69 | 1.41 | 1.02 | 0.86 | 1.07 |
| justoliunet: free (0 or ±2ᵏ) | 11 % | 12 % | 23 % | 30 % | 24 % |
| justoliunet_bn: adders/weight | 1.00 | 1.36 | 1.04 | 1.22 | 1.44 |

About one adder (~14 LUTs) per weight against ~50-65 LUTs for a general 8×8 LUT
multiplier. With the adder trees, the whole model hard-wired once is roughly
4 500 × ~28 ≈ 125 k LUTs (estimate).

**Rule: fold over positions and pixels, never over channels and taps.** Folding over
positions keeps each multiplier on one weight; folding over channels or taps sends the
weights through ROMs and multiplexers and rules this out (section 4 does that, and
should be revisited).

- ZCU104: conv3, conv4 and dense hard-wired in LUTs (~113 k LUTs), all DSPs on conv2
  and conv1: $T$ from 18 toward ~10, memory-bound, ~31 ms.
- Zynq-7030: the whole model does not fit in 79 k LUTs, conv3 alone (~36 k LUTs) does,
  at 18 cycles/pixel; the 400 DSPs for conv2 and conv4 (30 240 MACs): $T \approx$ 38-46,
  ~0.25-0.3 s instead of 0.53 s.
- Zynq-7020 (53 k LUTs): about one layer, e.g. conv2 (~12 k LUTs): ~0.9-1.0 s instead of
  1.35 s.

### B. Fast FIR algorithms (exact, fewer multiplies)

Along the spectrum each (input channel → output channel) pair is a 6-tap FIR filter.
The fast FIR algorithm computes several outputs together with integer pre- and
post-adds, so the result stays bit-exact (unlike Winograd, which needs fractions).
2-parallel: 9 multiplies per 2 outputs instead of 12 (−25 %); 3-parallel: 12 per 3
instead of 18 (−33 %); $k = 6$ divides by both. Pre-added weights are still constants
(combines with A). Pre-added activations are 9 bits: packed products just fit
(510 × 256 = 130 560 < 2¹⁷), with group 1. Largest gain where DSPs limit (7020/7030).

### C. Classify during the capture (HYPSO-2)

No neighbours are needed, so each pushbroom frame (1 092 pixels) can be classified as it
leaves the sensor, without storing the cube (the 2D model needs 32 lines buffered). At
$T = 81$ on the 7030 at 100 MHz a frame takes ~0.9 ms, against a frame period of ~50 ms
(exposure 47 ms); the 7020's ~1.8 ms also keeps up.

### D. With retraining (model and hardware co-design)

- 4-bit weights/activations (quantization-aware training): 4 multiplies per DSP or tiny
  hard-wired multipliers, potentially 2× again.
- Band selection: fewer input bands shorten every layer.
- Early exit: a tiny classifier settles easy pixels (open sea, thick cloud); only
  uncertain ones go to the full network. Data-dependent speed.

## 6. Next steps

1. Build and measure the current kernels (section 2's checklist; for the ZedBoard:
   `fpga_build justoliunet_opt_zynq7020 --board zedboard`, 216 DSPs expected, then on the
   board `classify cube.npy scores.npy reference_justoliunet.npy`, the reference from
   `python3 tools/justoliunet_model.py reference IMAGE`).
2. Prototype A on the ZCU104: conv3, conv4 and dense hard-wired (weights as compile-time
   constants, not loaded into registers), DSPs on conv2 and conv1. Its csynth report
   gives the real LUT cost per hard-wired weight, which every estimate in A depends on.
3. Decide the Zynq-7030 version from that number: conv3 hard-wired plus DSPs elsewhere,
   possibly with B on the DSP layers.

## Appendix: the tables in LaTeX

Needs `\usepackage{booktabs}`; compiled without errors or warnings (Tectonic 0.17).

```latex
\begin{table}[htbp]
  \centering
  \small
  \caption{ZCU104 (ZU7EV, $f = 200$\,MHz, 1\,728 DSPs, 230\,k LUTs), the implemented design:
    $T = 18$ cycles/pixel, $t = 653\,016 \cdot 18 / (200\,\mathrm{MHz}) = 58.8$\,ms,
    $\eta = 3188/3465 = 92$\,\%, 1\,304 of 1\,728 DSPs (75\,\%),
    $T_\mathrm{mem} = 240/(2 \cdot 16 \cdot 0.8) = 9.4$ cycles/pixel.}
  \label{tab:zcu104-layers}
  \begin{tabular}{lcccccc}
    \toprule
                         & conv1  & conv2   & conv3   & conv4   & dense & z-score \\
    \midrule
    $\mathit{POS} \cdot \mathit{OCP} \cdot \mathit{FP}$ & $6 \cdot 6 \cdot 6$ & $3 \cdot 12 \cdot 36$ & $1 \cdot 18 \cdot 72$ & $1 \cdot 6 \cdot 108$ & $3 \cdot 3$ & 8 lanes \\
    $m_l$                & 216    & 1\,296  & 1\,296  & 648     & 9     & 8 \\
    $S_l$ (cycles/pixel) & 17     & 16      & \textbf{18} & 16  & 16    & 15 \\
    Implementation       & LUT    & 648 DSP & 648 DSP & LUT     & LUT   & 8 DSP \\
    $u_l$                & 94\,\% & 85\,\%  & 100\,\% & 89\,\%  & 89\,\% & -- \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[htbp]
  \centering
  \small
  \caption{Zynq-7030 (HYPSO-2, $f = 100$\,MHz, 400 DSPs, 78.6\,k LUTs), DSP-only design:
    $T = 81$ cycles/pixel, $t = 653\,016 \cdot 81 / (100\,\mathrm{MHz}) \approx 0.53$\,s,
    364 of 400 DSPs (91\,\%), $\eta = 708/794 = 89$\,\%,
    $T_\mathrm{mem} = 240/(8 \cdot 0.8) = 37.5$ cycles/pixel with one 64-bit port.}
  \label{tab:z7030-layers}
  \begin{tabular}{lcccccc}
    \toprule
                         & conv1  & conv2   & conv3   & conv4   & dense & z-score \\
    \midrule
    $\mathit{POS} \cdot \mathit{OCP} \cdot \mathit{FP}$ & $2 \cdot 6 \cdot 6$ & $2 \cdot 4 \cdot 36$ & $2 \cdot 2 \cdot 72$ & $1 \cdot 4 \cdot 36$ & 2 & 4 lanes \\
    $m_l$                & 72     & 288     & 288     & 144     & 2     & 4 \\
    $S_l$ (cycles/pixel) & 51     & 69      & \textbf{81} & 72  & 72    & 30 \\
    Implementation       & LUT    & 144 DSP & 144 DSP & 72 DSP  & LUT   & 4 DSP \\
    $u_l$                & 63\,\% & 85\,\%  & 100\,\% & 89\,\%  & 89\,\% & -- \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[htbp]
  \centering
  \small
  \caption{Zynq-7020 (ZedBoard, $f = 78.1$\,MHz, 220 DSPs, 53.2\,k LUTs), DSP-only design:
    $T = 162$ cycles/pixel, $t = 653\,016 \cdot 162 / (78.1\,\mathrm{MHz}) \approx 1.35$\,s,
    about 184 of 220 DSPs (84\,\%), $\eta = 354/397 = 89$\,\%,
    about 1.5 bytes/cycle from DDR (one port is ample).}
  \label{tab:z7020-layers}
  \begin{tabular}{lccccc}
    \toprule
                         & conv1  & conv2   & conv3   & conv4   & dense \\
    \midrule
    $\mathit{POS} \cdot \mathit{OCP} \cdot \mathit{FP}$ & $1 \cdot 6 \cdot 6$ & $1 \cdot 4 \cdot 36$ & $1 \cdot 2 \cdot 72$ & $1 \cdot 2 \cdot 36$ & 1 \\
    $m_l$                & 36     & 144     & 144     & 72      & 1 \\
    $S_l$ (cycles/pixel) & 102    & 138     & \textbf{162} & 144 & 144 \\
    Implementation       & LUT    & 72 DSP  & 72 DSP  & 36 DSP  & LUT \\
    $u_l$                & 63\,\% & 85\,\%  & 100\,\% & 89\,\%  & 89\,\% \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[htbp]
  \centering
  \small
  \caption{Predicted inference time for a full $598 \times 1092$ capture ($N = 653\,016$ pixels).
    DSP only: DSP multipliers alone (conv1 and dense in LUTs). With LUT mult.: the LUT budget
    also filled with multipliers, $m_\mathrm{LUT} \approx (0.7\,L_\mathrm{chip} - L_\mathrm{infra})/c$,
    $c \approx 50$ LUTs per $8 \times 8$ multiplier. The ZCU104 DSP-only figure is the
    implemented design; all others are paper estimates. $^\ast$Memory-bound.}
  \label{tab:summary}
  \begin{tabular}{lccccc}
    \toprule
              &                  & \multicolumn{2}{c}{DSP only} & \multicolumn{2}{c}{With LUT mult.} \\
    \cmidrule(lr){3-4} \cmidrule(lr){5-6}
    Board     & $f_\mathrm{clk}$ & $T$ & Time              & $T$                 & Time \\
    \midrule
    ZCU104    & 200\,MHz         & 18  & \textbf{58.8\,ms} & $\approx 10^\ast$   & $\approx 31$\,ms \\
    Zynq-7030 & 100\,MHz         & 81  & \textbf{0.53\,s}  & $\approx 36$        & $\approx 0.24$\,s \\
    Zynq-7020 & 78.1\,MHz        & 162 & \textbf{1.35\,s}  & $\approx 66$        & $\approx 0.55$\,s \\
    \bottomrule
  \end{tabular}
\end{table}
```
