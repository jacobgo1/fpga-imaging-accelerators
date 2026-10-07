#ifndef JUSTOUNETSIMPLE_OPT_HLS_HPP
#define JUSTOUNETSIMPLE_OPT_HLS_HPP

#include <stdint.h>
#include "../../optimized/hls_vector_shim.hpp"

// The quantized 2D-JustoUNet (the same model as src/hls/justounetsimple/, the
// golden baseline) on a whole raw capture: one start classifies every pixel.
// The ARM only points the kernel at the cube in DDR; the kernel does what the
// host used to: drop bands, z-score, cut the image into 32 x 32 patches (the
// last row and column repeated to fill the edge patches), and put each patch's
// scores back in place. A DATAFLOW chain of processes, all running at once:
//
//   read_rows<0> ─┐                           even rows of each patch, HP0
//   read_rows<1> ─┴─► patch_buffer ─► conv1 ─► pool ─► conv2 ─► pool+up ─►
//                     z-score, ping-pong          conv3 ─► up ─► conv4 ─► store
//
// read_rows: a patch row is 32 pixels x 120 bands = 7.5 KB contiguous in the
// cube, one long burst. Two ports (each its own m_axi bundle and HP port,
// 128 bits per cycle) take alternate rows.
// patch_buffer: two patch buffers on chip (UltraRAM). While conv1 reads patch
// k from one, in band-chunk order, the readers fill patch k+1 into the other.
// conv1: 16 bands per cycle (7 x 32 x 32 + 33 cycles per patch); everything
// after it has time to spare and keeps up with fewer multipliers.
//
// Integer fixed point throughout (src/optimized/conv3x3_stream.hpp): int16
// activations, the int8 weights as they are, int32 sums. The z-score is in
// integers too (justounetsimple_prep.hpp, from mu_sd.txt), within one unit of
// the float one. tools/justounetsimple_model.py has the bit-exact numpy version
// (classify_fixed); on the aegean capture the scores are within 0.004 of the
// float model's, and 99.998% of the pixels get the same class.
//
// din0, din1: the same raw cube, [height][width][120] uint16 -- the two ports
// read different rows of it. The 120 bands of a pixel are 15 words of 8; words
// 1-14 are the kept bands (raw 8-117) plus raw 118-119, which the z-score
// zeroes: 7 chunks of 16 for conv1, no shuffling. Word 0 (raw 0-7) is dropped.
// dout: [height][width][3] int32 scores, score * 2^OUT_FRAC.
// Any height and width (>= 1). A strip of a larger image works the same way:
// point din at its first row and dout at that row's scores.

#define JOPT_H           32    // patch size
#define JOPT_W           32
#define JOPT_RAW_BANDS   120   // bands in the cube
#define JOPT_IN_CH       110   // bands the model uses: raw 8-117
#define JOPT_BASE_CH     6
#define JOPT_OUT_CH      3
#define JOPT_P1          16    // bands per cycle into conv1
#define JOPT_PORT_BANDS  8     // bands per 128-bit word
// IN_CH rounded up to a multiple of P1: 112 bands, the last 2 zero.
#define JOPT_IN_PAD      ((JOPT_IN_CH + JOPT_P1 - 1) / JOPT_P1 * JOPT_P1)

#define JOPT_IN_FRAC     11    // conv1's input: z-score * 2^11 (int16: -16 .. +16)
#define JOPT_ACT_FRAC    11    // activations between the blocks, the same
#define JOPT_OUT_FRAC    16    // dout = score * 2^16

// One word of the cube: 8 bands of a pixel, 128 bits.
typedef hls::vector<uint16_t, JOPT_PORT_BANDS> jopt_raw_t;

void justounetsimple_opt(const jopt_raw_t *din0, const jopt_raw_t *din1, int32_t *dout,
                         int height, int width);

#endif // JUSTOUNETSIMPLE_OPT_HLS_HPP
