#ifndef JUSTOLIUNET_OPT_ZYNQ7030_HLS_HPP
#define JUSTOLIUNET_OPT_ZYNQ7030_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// justoliunet_opt_zynq7020 (read its header first) sized for the Zynq-7030,
// HYPSO-2's FPGA: build it with --board zynq7030. The same model, formats and
// answers bit for bit; the same building blocks and ideas (folding, spectral
// streaming, two multiplies per DSP48E1, read-ahead), with the 7030's 400 DSPs
// (220 on the 7020), 79 k LUTs (53 k) and 100 MHz (78) spent on wider layers:
//
//   read_pixels_ahead ─► zscore ─► conv1 ─► pool ─► conv2 ─► conv3 ─► conv4 ─► dense ─► store
//          HP0            30       51      51      51       74       72      48       (HP1)
//   cycles a pixel, each process
//
//   layer   MACs/pixel  per cycle                  cycles   multipliers
//   conv1      3 672    2 pos x 6 out ch x 6           51     72 (12 x 8 bit)  LUTs
//   conv2     19 872    12 out ch x 6 in x 6           51    432, 2 per DSP    216 DSPs
//   conv3     23 328    18 out ch x 3 in x 6           74    324, 2 per DSP    162 DSPs
//   conv4     10 368    4 out ch x 6 in x 6            72    144 (8 x 8 bit)   LUTs
//   dense        144    1 feature x 3                  48      3               LUTs
//
//   378 of the 400 DSPs. conv3 sets the pace: 74 cycles a pixel (108 on the 7020),
//   about 0.48 s for a 598 x 1092 capture at 100 MHz (0.90 s on the 7020).
//
// What differs from the 7020's, and why:
// - conv1 computes two positions a cycle (conv1d_engine with POS 2; pool_adjacent
//   pools each pair as it comes): a streamed layer takes in at most one position
//   a cycle, and conv1's 102 positions were the 7020's floor.
// - conv2 does all its channels at once (one step an output position), so it takes
//   its 51 input positions in 51 cycles; with two steps it would be 96, nothing in
//   between. conv3 takes 4 steps an output (18 x 4 + 2 fill = 74); conv4 18
//   (4 x 18 = 72, its fill hidden in the read-ahead).
// - The DSPs go to conv2 and conv3 (75 % of the work), as on the 7020; the 22
//   left over cannot hold conv1 (72 multipliers) or conv4 (144).
// One read port is still plenty: 30 cycles a pixel against 74.
//
// din0: the raw cube, [height][width][120] uint16. dout: [height][width][4]
// int32: the 3 scores (score * 2^16), then the class (0 cloud, 1 land, 2 sea).
// cycles: the kernel's own count of its clock cycles. Any height and width from
// 1 to 65535.

constexpr int RAW_BANDS = 120;   // bands in the cube
constexpr int PORT_BANDS = 4;    // bands per 64-bit word
constexpr int WORDS = RAW_BANDS / PORT_BANDS;   // 30 words a pixel
constexpr int FIRST = 2;         // words 0-1 (raw bands 0-7) are dropped
constexpr int KEPT = WORDS - FIRST;             // words 2-29: kept bands 0-111
constexpr int BANDS = KEPT * PORT_BANDS;        // 112 (110 and 111 are zero)
constexpr int K = 6;             // every conv's kernel
constexpr int CLASSES = 3;
constexpr int PORTS = 1;         // one read port is plenty
constexpr int BLOCK = 8;         // pixels a burst: 240 words

// The layers' folding and where their multipliers go (see above).
constexpr int C1_POS = 2;
constexpr int C2_OCP = 12, C2_ICP = 6;  constexpr bool C2_PACKED = true;
constexpr int C3_OCP = 18, C3_ICP = 3;  constexpr bool C3_PACKED = true;
constexpr int C4_OCP = 4, C4_ICP = 6;   constexpr bool C4_PACKED = false;
constexpr int DENSE_FEAT = 1;
constexpr bool ZSCORE_DSP = false;

// One word of the cube: 4 bands of a pixel, 64 bits.
typedef hls::vector<uint16_t, PORT_BANDS> jl_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> jl_out_t;

void justoliunet_opt_zynq7030(hls::burst_maxi<jl_raw_t> din0, hls::burst_maxi<jl_out_t> dout,
                              int height, int width, uint32_t& cycles);

#endif // JUSTOLIUNET_OPT_ZYNQ7030_HLS_HPP
