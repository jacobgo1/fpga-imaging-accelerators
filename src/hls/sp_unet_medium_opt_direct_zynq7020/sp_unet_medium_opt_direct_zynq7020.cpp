// sp_unet_medium_opt_direct_zynq7020: the direct SpecPW-UNet, medium, on a whole raw
// capture, sized for the Zynq-7020. The design is in sp_unet_opt_zynq7020.hpp, the
// processes in sp_unet_direct_zynq7020_dataflow.hpp; here are its weights and the
// size of every process.
#include "sp_unet_medium_opt_direct_zynq7020.hpp"

// The quantized weights, as the int8 integers q they are (value q * 2^-frac).
#include "../../../weights/quantized/sp_unet_medium/sp_unet_medium_int8.hpp"
namespace q = sp_unet_medium_int8;

// The DSPs set the pace: every process takes at most T = 228 312 cycles a patch,
// about 1 946 ms for a 598 x 1092 capture at 78.1 MHz (the board's 12.8 ns). A conv
// layer takes (H x W + W + 1) x IN/PI x OUT/PO x 3/KR cycles a patch
// (conv3x3_grouped), with 3 x KR x PI x PO multipliers; the table is the
// fastest that fits 205 DSPs.
namespace cfg {
constexpr int P1 = 4;               // convsp1: 4 bands a chunk, 28 chunks
constexpr int SP1_PASSES = 6;       // ... x 6 passes over its pairs: 172 032 cycles a patch, 8 DSPs
constexpr bool ZSCORE_DSP = false;  // the z-score's 4 multipliers in LUTs
constexpr int SP2_PI = 2, SP2_PO = 1;  // convsp2 24 -> 16: 12 x 16 = 192 cycles a pixel (196 608), 2 DSPs
constexpr bool SP2_DSP = true;
constexpr int SP3_PI = 1, SP3_PO = 1;  // convsp3 16 -> 12: 16 x 12 = 192 cycles a pixel (196 608), 1 DSPs
constexpr bool SP3_DSP = true;
constexpr int FIN_PI = 1, FIN_PO = 1;  // final 12 -> 3: 12 x 3 = 36 cycles a pixel (36 864), 1 DSP
constexpr bool FIN_DSP = true;
//                       PI  PO KR DSP           channels  cycles a pixel          a patch    multipliers
typedef spu7_layer< 4, 1, 1, true> D1A;  // 12 -> 24   3 x 24 x 3 =  216 a pixel  x 1057 = 228 312   12
typedef spu7_layer< 8, 1, 1, true> D1B;  // 24 -> 24   3 x 24 x 3 =  216 a pixel  x 1057 = 228 312   24
typedef spu7_layer< 6, 1, 1, true> D2A;  // 24 -> 48   4 x 48 x 3 =  576 a pixel  x  273 = 157 248   18
typedef spu7_layer< 3, 3, 1, true> D2B;  // 48 -> 48  16 x 16 x 3 =  768 a pixel  x  273 = 209 664   27
typedef spu7_layer< 3, 3, 1, true> UP1;  // 48 -> 48  16 x 16 x 3 =  768 a pixel  x  273 = 209 664   27
typedef spu7_layer< 6, 1, 1, true> D3A;  // 48 -> 24   8 x 24 x 3 =  576 a pixel  x  273 = 157 248   18
typedef spu7_layer< 3, 1, 1, true> D3B;  // 24 -> 24   8 x 24 x 3 =  576 a pixel  x  273 = 157 248    9
typedef spu7_layer< 8, 1, 1, true> UP2;  // 24 -> 24   3 x 24 x 3 =  216 a pixel  x 1057 = 228 312   24
typedef spu7_layer< 8, 1, 1, true> D4A;  // 48 -> 12   6 x 12 x 3 =  216 a pixel  x 1057 = 228 312   24
typedef spu7_layer< 2, 1, 1, true> D4B;  // 12 -> 12   6 x 12 x 3 =  216 a pixel  x 1057 = 228 312    6
// DSPs: 189 convs + 8 convsp1 + 4 convsp2/3 and final = 201 of 220 (the budget: 205).
}

#include "../sp_unet_small_opt_direct_zynq7020/sp_unet_direct_zynq7020_dataflow.hpp"

void sp_unet_medium_opt_direct_zynq7020(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_out_t> dout,
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
