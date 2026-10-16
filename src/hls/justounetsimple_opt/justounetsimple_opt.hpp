#ifndef JUSTOUNETSIMPLE_OPT_HLS_HPP
#define JUSTOUNETSIMPLE_OPT_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// The quantized 2D-JustoUNet (the same model as src/hls/justounetsimple/, the
// golden baseline) on a whole raw capture: one start classifies every pixel.
// The ARM only points the kernel at the cube in DDR; the kernel does what the
// host used to: drop bands, z-score, cut the image into 32 x 32 patches (the
// last row and column repeated to fill the edge patches), and put each patch's
// scores back in place. A DATAFLOW chain of processes, all running at once:
//
//   read_rows_ahead<0> ─┐                     rows 0, 4, 8, ... of each patch, HP0
//   read_rows_ahead<1> ─┤
//   read_rows_ahead<2> ─┤
//   read_rows_ahead<3> ─┴─► patch_buffer ─► conv1 ─► pool ─► conv2 ─► pool+up ─►
//                     z-score, ping-pong          conv3 ─► up ─► conv4 ─► store (HPC0)
//
// read_rows_ahead: a patch row is 32 pixels x 120 bands = 7.5 KB contiguous
// in the cube, two bursts of 16 pixels. Four ports (each its own m_axi bundle
// and HP port, 128 bits per cycle) take every fourth row, each asking for its
// next row while the current one still comes in, so the DDR's latency is not
// paid once per row.
// patch_buffer: two patch buffers on chip (UltraRAM). While conv1 reads patch
// k from one, in band-chunk order, the readers fill patch k+1 into the other.
// conv1: 32 bands per cycle (4 x 32 x 32 + 33 cycles per patch), two
// multiplies per DSP; everything after it has time to spare and keeps up with
// fewer multipliers.
//
// Integer fixed point throughout (src/optimized/): int8 input to conv1, int16
// activations after it, the int8 weights as they are, int32 sums. The z-score is
// in integers too (justounetsimple_prep.hpp, from mu_sd.txt), within one unit of
// the float one. tools/justounetsimple_model.py has the bit-exact numpy version
// (classify_fixed). The int8 input moves the scores by up to ~1 on a few pixels
// against the float model, but on the aegean capture 99.96% of the pixels get
// the same class and the accuracy against the labels is the same.
//
// din0 .. din3: the same raw cube, [height][width][120] uint16 -- the four ports
// read different rows of it. The 120 bands of a pixel are 15 words of 8; words
// 1-14 are the kept bands (raw 8-117) plus raw 118-119, which the z-score
// zeroes: four words per chunk of conv1, no shuffling (the last chunk has two,
// and zeros). Word 0 (raw 0-7) is dropped.
// dout: [height][width][4] int32: the 3 scores (score * 2^OUT_FRAC), then the
// class (0 cloud, 1 land, 2 sea; the first highest score). Four, not three, so
// that a pixel is one 128-bit word: one write per pixel, not three.
// Any height and width from 1 to 65535 (the index arithmetic is 16-bit). A
// strip of a larger image works the same way: point din at its first row and
// dout at that row's scores.
// cycles: an output register, the kernel's time in clock cycles, from its
// start until the last score is in memory. 0 in C simulation, where the
// processes run one after another.
// trace: the timeline of the run, TRACE_WORDS uint32, through its own port
// (HPC1); src/optimized/shared/timeline.hpp has the layout. Each FIFO below
// has two lanes: "gave", lane F, where the stage before it writes, and
// "took", lane 12 + F, where the stage after it reads. For each patch, the
// cycle of its first and its last word there; for the whole run, the cycles
// the tap found the FIFO before it empty and the one after it full.
//   F 0-3  read_rows_ahead 0-3 -> patch_buffer     3 840 words a patch each
//   F 4    patch_buffer -> conv1                   4 096
//   F 5    conv1 -> relu_pool                      1 024
//   F 6    relu_pool -> conv2                        256
//   F 7    conv2 -> relu_pool_up                     256
//   F 8    relu_pool_up -> conv3                     256
//   F 9    conv3 -> relu_up                          256
//   F 10   relu_up -> conv4                        1 024
//   F 11   conv4 -> store_scores                   1 024
// The first TRACE_PATCHES patches are kept (an image of 19 x 35 has 665).

// The kernel's sizes and formats, used as they are by the .cpp and the testbench.
// constexpr, not #define: a macro H would also rewrite every template
// parameter called H in src/optimized/; a constant is just hidden by it.
constexpr int H = 32, W = 32;    // patch size
constexpr int RAW_BANDS = 120;   // bands in the cube
constexpr int IN_CH = 110;       // bands the model uses: raw 8-117
constexpr int BASE_CH = 6;
constexpr int OUT_CH = 3;
constexpr int PORTS = 4;         // read ports, each a quarter of the rows
constexpr int P1 = 32;           // bands per cycle into conv1
constexpr int PORT_BANDS = 8;    // bands per 128-bit word
// IN_CH rounded up to a multiple of P1: 128 bands, the last 18 zero.
constexpr int IN_PAD = (IN_CH + P1 - 1) / P1 * P1;

constexpr int IN_FRAC = 5;       // conv1's input: z-score * 2^5 (int8: -3.97 .. +3.97)
constexpr int ACT_FRAC = 11;     // activations between the blocks (int16: -16 .. +16)
constexpr int OUT_FRAC = 16;     // dout = score * 2^16

constexpr int TRACE_FIFOS = 12;
constexpr int TRACE_LANES = 2 * TRACE_FIFOS;   // gave 0-11, took 12-23
constexpr int TRACE_PATCHES = 1024;
constexpr int TRACE_HEAD = 128;  // the total and the lanes' counts, then the lanes
constexpr int TRACE_WORDS = TRACE_HEAD + TRACE_LANES * 2 * TRACE_PATCHES;

// One word of the cube: 8 bands of a pixel, 128 bits.
typedef hls::vector<uint16_t, PORT_BANDS> jopt_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> jopt_out_t;

void justounetsimple_opt(hls::burst_maxi<jopt_raw_t> din0, hls::burst_maxi<jopt_raw_t> din1,
                         hls::burst_maxi<jopt_raw_t> din2, hls::burst_maxi<jopt_raw_t> din3,
                         hls::burst_maxi<jopt_out_t> dout, int height, int width, uint32_t& cycles,
                         uint32_t *trace);

#endif // JUSTOUNETSIMPLE_OPT_HLS_HPP
