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
//   read_rows<0> ─┐                           rows 0, 4, 8, ... of each patch, HP0
//   read_rows<1> ─┤
//   read_rows<2> ─┤
//   read_rows<3> ─┴─► patch_buffer ─► conv1 ─► pool ─► conv2 ─► pool+up ─►
//                     z-score, ping-pong          conv3 ─► up ─► conv4 ─► store (HPC0)
//
// read_rows: a patch row is 32 pixels x 120 bands = 7.5 KB contiguous in the
// cube, one long burst. Four ports (each its own m_axi bundle and HP port,
// 128 bits per cycle) take every fourth row.
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
// that a pixel is one 128-bit word.
// Any height and width (>= 1). A strip of a larger image works the same way:
// point din at its first row and dout at that row's scores.

#define JOPT_H           32    // patch size
#define JOPT_W           32
#define JOPT_RAW_BANDS   120   // bands in the cube
#define JOPT_IN_CH       110   // bands the model uses: raw 8-117
#define JOPT_BASE_CH     6
#define JOPT_OUT_CH      3
#define JOPT_PORTS       4     // read ports, each a quarter of the rows
#define JOPT_P1          32    // bands per cycle into conv1
#define JOPT_PORT_BANDS  8     // bands per 128-bit word
// IN_CH rounded up to a multiple of P1: 128 bands, the last 18 zero.
#define JOPT_IN_PAD      ((JOPT_IN_CH + JOPT_P1 - 1) / JOPT_P1 * JOPT_P1)

#define JOPT_IN_FRAC     5     // conv1's input: z-score * 2^5 (int8: -3.97 .. +3.97)
#define JOPT_ACT_FRAC    11    // activations between the blocks (int16: -16 .. +16)
#define JOPT_OUT_FRAC    16    // dout = score * 2^16

// One word of the cube: 8 bands of a pixel, 128 bits.
typedef hls::vector<uint16_t, JOPT_PORT_BANDS> jopt_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> jopt_out_t;

void justounetsimple_opt(const jopt_raw_t *din0, const jopt_raw_t *din1,
                         const jopt_raw_t *din2, const jopt_raw_t *din3,
                         jopt_out_t *dout, int height, int width);

#endif // JUSTOUNETSIMPLE_OPT_HLS_HPP
