// sp_unet_large_opt_direct_zynq7020: the direct SpecPW-UNet, large, on a whole raw
// capture, sized for the Zynq-7020. The design is in sp_unet_opt_zynq7020.hpp, the
// processes in sp_unet_direct_zynq7020_dataflow.hpp; here are its weights and the
// size of every process.
#include "sp_unet_large_opt_direct_zynq7020.hpp"

// The quantized weights, as the int8 integers q they are (value q * 2^-frac).
#include "../../../weights/quantized/sp_unet_large/sp_unet_large_int8.hpp"
namespace q = sp_unet_large_int8;

// The DSPs set the pace: every process takes at most T = 420 864 cycles a patch,
// about 3 588 ms for a 598 x 1092 capture at 78.1 MHz (the board's 12.8 ns). A conv
// layer takes (H x W + W + 2) x 9/KT x IN/PI x OUT/PO cycles a patch
// (conv3x3_lines), with KT x PI x PO multipliers; the table is the fastest
// that fits 205 DSPs.
namespace cfg {
constexpr int P1 = 4;               // convsp1: 4 bands a chunk, 28 chunks
constexpr int SP1_PASSES = 8;       // ... x 8 passes over its pairs: 229 376 cycles a patch, 8 DSPs
constexpr int SKIP_DEPTH = 1024;    // the skip FIFO, pixels (a patch: the block RAM is short)
constexpr bool ZSCORE_DSP = true;  // the z-score's 4 multipliers on DSPs
constexpr int SP2_PI = 2, SP2_PO = 1;  // convsp2 32 -> 16: 16 x 16 = 256 cycles a pixel (262 144), 2 DSPs
constexpr bool SP2_DSP = true;
constexpr int SP3_PI = 1, SP3_PO = 1;  // convsp3 16 -> 16: 16 x 16 = 256 cycles a pixel (262 144), 1 DSPs
constexpr bool SP3_DSP = true;
constexpr int FIN_PI = 1, FIN_PO = 1;  // final 16 -> 3: 16 x 3 = 48 cycles a pixel (49 152), 1 DSP
constexpr bool FIN_DSP = true;
//                       PI  PO KT DSP           channels  taps x in x out groups      a patch    multipliers
typedef spu7_layer< 4, 1, 3, true> D1A;  // 16 -> 32  3 x  4 x 32 =  384 a pixel  x 1058 = 406 272   12
typedef spu7_layer< 8, 1, 3, true> D1B;  // 32 -> 32  3 x  4 x 32 =  384 a pixel  x 1058 = 406 272   24
typedef spu7_layer< 4, 1, 3, true> D2A;  // 32 -> 64  3 x  8 x 64 = 1536 a pixel  x  274 = 420 864   12
typedef spu7_layer< 8, 1, 3, true> D2B;  // 64 -> 64  3 x  8 x 64 = 1536 a pixel  x  274 = 420 864   24
typedef spu7_layer< 8, 1, 3, true> UP1;  // 64 -> 64  3 x  8 x 64 = 1536 a pixel  x  274 = 420 864   24
typedef spu7_layer< 4, 1, 3, true> D3A;  // 64 -> 32  3 x 16 x 32 = 1536 a pixel  x  274 = 420 864   12
typedef spu7_layer< 2, 1, 3, true> D3B;  // 32 -> 32  3 x 16 x 32 = 1536 a pixel  x  274 = 420 864    6
typedef spu7_layer< 8, 1, 3, true> UP2;  // 32 -> 32  3 x  4 x 32 =  384 a pixel  x 1058 = 406 272   24
typedef spu7_layer< 8, 1, 3, true> D4A;  // 64 -> 16  3 x  8 x 16 =  384 a pixel  x 1058 = 406 272   24
typedef spu7_layer< 2, 1, 3, true> D4B;  // 16 -> 16  3 x  8 x 16 =  384 a pixel  x 1058 = 406 272    6
// DSPs: 168 convs + 8 convsp1 + 4 convsp2/3 and final + 4 z-score = 184 of 220 (the budget: 205).
}

#include "../sp_unet_small_opt_direct_zynq7020/sp_unet_direct_zynq7020_dataflow.hpp"

void sp_unet_large_opt_direct_zynq7020(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_out_t> dout,
                                       int height, int width, uint32_t& cycles)
{
// Two AXI masters, each on its own 64-bit HP port to DDR (boards/zynq7020/system.tcl):
// one reads the cube through HP0, one writes the scores through HP1.
// depth is only for co-simulation: the testbench's 40 x 50 image.
#pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=60000 max_read_burst_length=256 num_read_outstanding=4
#pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem1 offset=slave depth=2000 max_write_burst_length=32 num_write_outstanding=16
#pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=height bundle=control
#pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
#pragma HLS INTERFACE mode=s_axilite port=cycles bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    sp_unet_direct_zynq7020_dataflow(din0, dout, height, width, cycles);
}
