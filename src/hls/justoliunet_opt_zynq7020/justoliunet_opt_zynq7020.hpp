#ifndef JUSTOLIUNET_OPT_ZYNQ7020_HLS_HPP
#define JUSTOLIUNET_OPT_ZYNQ7020_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// justoliunet_opt (read its header first) sized for the Zynq-7020, the lab's
// ZedBoards: build it with --board zedboard. The same model, the same integer
// formats, the same answers bit for bit, the same raw cube in and scores out;
// what changes is what fits on the chip (220 DSP48E1s, 53 k LUTs, 64-bit HP
// ports, 78 MHz) and so how the work is spread over time:
//
//   read_pixels_ahead ─► zscore ─► conv1 ─► pool ─► conv2 ─► conv3 ─► conv4 ─► dense ─► store
//          HP0            30      102     102      96      108      96      48       (HP1)
//   cycles a pixel, each process; all of them run at once, a pixel each apart
//
// The same ideas as on the ZCU104: every layer about the same number of cycles a
// pixel, so no stage waits; the DSPs where the work is (conv2 and conv3, 75 %
// of it), two multiplies each; uint8 activations; one long burst per block of
// pixels, requested ahead. What differs:
//
// - Folding. 220 DSPs cannot compute a layer's positions side by side, so each
//   layer takes one output position at a time and spreads it over steps: OCP of
//   its output channels and ICP of its input channels a cycle, all 6 taps, the
//   input-channel chunks summed in an accumulator (conv1d_stream.hpp).
//
//   layer   MACs/pixel  per cycle                  cycles   multipliers
//   conv1      3 672    6 out ch x 1 in x 6           102     36 (12 x 8 bit)  LUTs
//   conv2     19 872    6 out ch x 6 in x 6            96    216, 2 per DSP    108 DSPs
//   conv3     23 328    6 out ch x 6 in x 6           108    216, 2 per DSP    108 DSPs
//   conv4     10 368    2 out ch x 9 in x 6            96    108 (8 x 8 bit)   LUTs
//   dense        144    1 feature x 3                  48      3               LUTs
//
//   216 of the 220 DSPs; the z-score's 4 multipliers are in LUTs too. conv3 sets
//   the pace: 108 cycles a pixel, about 0.90 s for a 598 x 1092 capture at
//   78.1 MHz.
// - Spectral streaming. From conv1 on, a layer's input arrives one position (all
//   its channels) at a time and the layer keeps a window of the last 6, as a 1D
//   line buffer: narrow FIFOs and small windows instead of whole feature maps,
//   which the 7020's LUTs could not hold next to the multipliers. A layer spends
//   5 cycles a pixel filling its window, but it reads the next pixel's first
//   positions during the steps of its last output that read nothing, so only
//   conv2 (one such step) still spends 4 (conv3: 18 x 3 x 2 = 108, no fill).
// - Packing on the DSP48E1 (25 x 18 bits): the two weights 16 bits apart, and a
//   uint8 product (up to 32 640) fills those 16 bits, so every product is split
//   on its own (GROUP 1, conv1d_stream.hpp). int8 activations would allow 2 a
//   group, but would change the answers; these are the ZCU104's, bit for bit.
// - One read port: a pixel is 30 words of 4 bands (64-bit ports), 30 cycles at
//   one word a cycle, far below 108. Blocks of 8 pixels, 240 words a burst.
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
constexpr int C2_OCP = 6, C2_ICP = 6;   constexpr bool C2_PACKED = true;
constexpr int C3_OCP = 6, C3_ICP = 6;   constexpr bool C3_PACKED = true;
constexpr int C4_OCP = 2, C4_ICP = 9;   constexpr bool C4_PACKED = false;
constexpr int DENSE_FEAT = 1;
constexpr bool ZSCORE_DSP = false;

// One word of the cube: 4 bands of a pixel, 64 bits.
typedef hls::vector<uint16_t, PORT_BANDS> jl_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> jl_out_t;

void justoliunet_opt_zynq7020(hls::burst_maxi<jl_raw_t> din0, hls::burst_maxi<jl_out_t> dout,
                              int height, int width, uint32_t& cycles);

#endif // JUSTOLIUNET_OPT_ZYNQ7020_HLS_HPP
