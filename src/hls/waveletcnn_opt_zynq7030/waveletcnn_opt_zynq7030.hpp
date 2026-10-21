#ifndef WAVELETCNN_OPT_ZYNQ7030_HLS_HPP
#define WAVELETCNN_OPT_ZYNQ7030_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// waveletcnn_opt (read its header first) sized for the Zynq-7030, HYPSO-2's
// FPGA: build it with --board zynq7030. The same model, the same integer
// arithmetic, the same answers bit for bit, the same raw cube in and scores
// out, sized for the 7030's 400 DSPs, 265 BRAM36, four 64-bit HP ports and
// 100 MHz clock:
//
//   read_rows_ahead<0..3> ─► pixel_buffer ─► dwt_proj ─► conv_block ─► leaky_head ─► store (HP3)
//   HP0-HP3 (4 ports)        z-score, BRAM   1D DWT +    3x3, 12->12   LeakyReLU +
//                                            1x1 proj                  1x1 head
//
// - dwt_proj runs M_PAR = 4 subband steps per cycle (8 bands/cycle, 15 cycles
//   per pixel = 15 360 cycles/patch): 4 x 4 x 12 = 192 DSPs for the 226 -> 12
//   projection, and the 8-tap constant DWT filter in LUTs.
// - conv_block (12 -> 12 3x3) runs conv3x3_narrow at 1 output channel/cycle
//   (12 cycles/pixel = 12 684 cycles/patch): 12 x 9 = 108 DSPs.
// - leaky_head (LeakyReLU + 12 -> 3 1x1 seg_head + LeakyReLU) runs at 1 cycle
//   per pixel (1 024 cycles/patch): 12 x 3 = 36 DSPs.
// - Total DSPs: 192 + 108 + 36 = 336 of 400 (the 16 z-score multipliers are in
//   LUTs).
// - Four 64-bit read ports (HP0-HP3, 7 680 cycles/patch); the writer shares HP3
//   with the fourth reader (boards/zynq7030/system.tcl).
// dwt_proj sets the pace: 15 360 cycles/patch, 0.154 ms at 100 MHz, about
// 102 ms for a 598 x 1092 capture.

constexpr int H = 32, W = 32;        // patch size
constexpr int RAW_BANDS = 120;       // bands in the cube
constexpr int IN_CH = 110;           // bands the model uses: raw 8-117
constexpr int DWT_LEN = 8;           // Daubechies-4 filter length
constexpr int SUB_CH = (IN_CH + DWT_LEN - 1) / 2;   // 58 per subband
constexpr int CAT_CH = IN_CH + 2 * SUB_CH;          // 226 into projection
constexpr int MID_CH = 12;           // channels after projection and conv_block
constexpr int OUT_CH = 3;
constexpr int PORTS = 4;             // read ports, each a quarter of the rows
constexpr int M_PAR = 4;             // DWT subband steps per cycle (15 cycles/pixel)
constexpr int P1 = 2 * M_PAR;        // 8 bands per cycle into dwt_proj
constexpr int DWT_STEPS = (SUB_CH + M_PAR - 1) / M_PAR;   // 15 cycles per pixel
constexpr int PORT_BANDS = 4;        // bands per 64-bit word

constexpr int IN_FRAC = 5;           // dwt_proj's input: z-score * 2^5 (int8: -3.97 .. +3.97)
constexpr int ACT_FRAC = 9;          // activations between the blocks (int16: -64 .. +64)
constexpr int OUT_FRAC = 15;         // dout = score * 2^15

// One word of the cube: 4 bands of a pixel, 64 bits.
typedef hls::vector<uint16_t, PORT_BANDS> wopt_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> wopt_out_t;

void waveletcnn_opt_zynq7030(hls::burst_maxi<wopt_raw_t> din0, hls::burst_maxi<wopt_raw_t> din1,
                             hls::burst_maxi<wopt_raw_t> din2, hls::burst_maxi<wopt_raw_t> din3,
                             hls::burst_maxi<wopt_out_t> dout, int height, int width);

#endif // WAVELETCNN_OPT_ZYNQ7030_HLS_HPP
