#ifndef WAVELETCNN_OPT_ZYNQ7020_HLS_HPP
#define WAVELETCNN_OPT_ZYNQ7020_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// waveletcnn_opt (read its header first) sized for the Zynq-7020 (the PYNQ-Z2
// and ZedBoard): build it with --board zynq7020 or --board zedboard. The same
// model, the same integer arithmetic, the same answers bit for bit, the same
// raw cube in and scores out, sized for the 7020's 220 DSPs, 140 BRAM36 and
// two 64-bit read ports:
//
//   read_rows_ahead<0> ─┬─► pixel_buffer ─► dwt_proj ─► conv_block ─► leaky_head ─► store (HP2)
//   read_rows_ahead<1> ─┘   z-score, BRAM   1D DWT +    3x3, 12->12   LeakyReLU +
//                                           1x1 proj                  1x1 head
//
// - dwt_proj runs M_PAR = 2 subband steps per cycle (4 bands/cycle, 29 cycles
//   per pixel = 29 696 cycles/patch): 2 x 4 x 12 = 96 DSPs for the 226 -> 12
//   projection, and the 8-tap constant DWT filter in LUTs.
// - conv_block (12 -> 12 3x3) runs conv3x3_serial at P2 = 6 input x 1 output
//   channel/cycle (24 cycles/pixel = 25 368 cycles/patch): 6 x 9 = 54 DSPs.
// - leaky_head (LeakyReLU + 12 -> 3 1x1 seg_head + LeakyReLU) runs at 1 cycle
//   per pixel (1 024 cycles/patch): 12 x 3 = 36 DSPs.
// - Total DSPs: 96 + 54 + 36 = 186 of 220 (the 8 z-score multipliers are in
//   LUTs).
// - Two 64-bit read ports (HP0, HP1, 15 360 cycles/patch); the scores go out
//   through HP2.
// dwt_proj sets the pace: 29 696 cycles/patch, 0.38 ms at 78.1 MHz, about
// 253 ms for a 598 x 1092 capture.

constexpr int H = 32, W = 32;        // patch size
constexpr int RAW_BANDS = 120;       // bands in the cube
constexpr int IN_CH = 110;           // bands the model uses: raw 8-117
constexpr int DWT_LEN = 8;           // Daubechies-4 filter length
constexpr int SUB_CH = (IN_CH + DWT_LEN - 1) / 2;   // 58 per subband
constexpr int CAT_CH = IN_CH + 2 * SUB_CH;          // 226 into projection
constexpr int MID_CH = 12;           // channels after projection and conv_block
constexpr int OUT_CH = 3;
constexpr int PORTS = 2;             // read ports, each half of the rows
constexpr int M_PAR = 2;             // DWT subband steps per cycle (29 cycles/pixel)
constexpr int P1 = 2 * M_PAR;        // 4 bands per cycle into dwt_proj
constexpr int DWT_STEPS = (SUB_CH + M_PAR - 1) / M_PAR;   // 29 cycles per pixel
constexpr int P2 = 6;                // input channels per cycle in conv_block (24 cycles/pixel)
constexpr int PORT_BANDS = 4;        // bands per 64-bit word

constexpr int IN_FRAC = 5;           // dwt_proj's input: z-score * 2^5 (int8: -3.97 .. +3.97)
constexpr int ACT_FRAC = 9;          // activations between the blocks (int16: -64 .. +64)
constexpr int OUT_FRAC = 15;         // dout = score * 2^15

// One word of the cube: 4 bands of a pixel, 64 bits.
typedef hls::vector<uint16_t, PORT_BANDS> wopt_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> wopt_out_t;

void waveletcnn_opt_zynq7020(hls::burst_maxi<wopt_raw_t> din0, hls::burst_maxi<wopt_raw_t> din1,
                             hls::burst_maxi<wopt_out_t> dout, int height, int width);

#endif // WAVELETCNN_OPT_ZYNQ7020_HLS_HPP
