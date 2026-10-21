#ifndef WAVELETCNN_OPT_HLS_HPP
#define WAVELETCNN_OPT_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// The quantized WaveletCNN (the same model as src/hls/waveletcnn/, the golden
// baseline) on a whole raw capture: one start classifies every pixel.
// The ARM only points the kernel at the cube in DDR; the kernel drops bands,
// z-scores, cuts the image into 32 x 32 patches (the last row and column
// repeated to fill the edge patches), and writes each patch's scores back in
// place. A DATAFLOW chain of processes, all running at once:
//
//   read_rows_ahead<0> ─┐                     rows 0, 4, 8, ... of each patch, HP0
//   read_rows_ahead<1> ─┤
//   read_rows_ahead<2> ─┤
//   read_rows_ahead<3> ─┴─► pixel_buffer ─► dwt_proj ─► conv_block ─► leaky_head ─► store (HPC0)
//                     z-score, 4-row BRAM    1D DWT +    3x3, 12->12   LeakyReLU +
//                     strip ping-pong        1x1 proj                  1x1 head
//
// Because WaveletCNN's first two layers (dwt_spectral and the 1x1 projection
// 226 -> 12) are purely spectral / pointwise per pixel, two structural
// simplifications happen compared to a 3x3-first network:
// - pixel_buffer only needs a ping-pong buffer of one 4-row strip (4 x 32
//   pixels, 8x smaller than a full patch, in block RAM) to reorder the four
//   row readers into raster pixel order.
// - dwt_proj fuses the 8-tap stride-2 Daubechies-4 DWT with the 226 -> 12 1x1
//   projection: an 8-band sliding window computes low[m] and high[m] on the fly
//   and accumulates their products -- and x[2m], x[2m + 1]'s -- straight into
//   the 12 output channels, so the 226-channel intermediate tensor is never
//   stored.
//
// Integer fixed point throughout (src/optimized/): int8 input to dwt_proj,
// exact 12-fraction-bit DWT products, int16 activations after projection and
// conv_block, int8 weights as they are, int32 sums. tools/waveletcnn_model.py
// has the bit-exact numpy version (classify_fixed).

constexpr int H = 32, W = 32;        // patch size
constexpr int RAW_BANDS = 120;       // bands in the cube
constexpr int IN_CH = 110;           // bands the model uses: raw 8-117
constexpr int DWT_LEN = 8;           // Daubechies-4 filter length
constexpr int SUB_CH = (IN_CH + DWT_LEN - 1) / 2;   // 58 per subband
constexpr int CAT_CH = IN_CH + 2 * SUB_CH;          // 226 into projection
constexpr int MID_CH = 12;           // channels after projection and conv_block
constexpr int OUT_CH = 3;
constexpr int PORTS = 4;             // read ports, each a quarter of the rows
constexpr int M_PAR = 8;             // DWT subband steps per cycle (8 steps = 8 cycles/pixel)
constexpr int P1 = 2 * M_PAR;        // 16 bands per cycle into dwt_proj
constexpr int DWT_STEPS = (SUB_CH + M_PAR - 1) / M_PAR;   // 8 cycles per pixel
constexpr int CONV_STEPS = 6;        // 6 cycles per pixel in conv_block (2 output channels/cycle)
constexpr int PORT_BANDS = 8;        // bands per 128-bit word

constexpr int IN_FRAC = 5;           // dwt_proj's input: z-score * 2^5 (int8: -3.97 .. +3.97)
constexpr int ACT_FRAC = 9;          // activations between the blocks (int16: -64 .. +64)
constexpr int OUT_FRAC = 15;         // dout = score * 2^15

// One word of the cube: 8 bands of a pixel, 128 bits.
typedef hls::vector<uint16_t, PORT_BANDS> wopt_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> wopt_out_t;

void waveletcnn_opt(hls::burst_maxi<wopt_raw_t> din0, hls::burst_maxi<wopt_raw_t> din1,
                    hls::burst_maxi<wopt_raw_t> din2, hls::burst_maxi<wopt_raw_t> din3,
                    hls::burst_maxi<wopt_out_t> dout, int height, int width, uint32_t& cycles);

#endif // WAVELETCNN_OPT_HLS_HPP
