// sp_unet_small_opt_direct_zynq7020: the direct SpecPW-UNet, small, on a whole raw
// capture, sized for the Zynq-7020. The design is in sp_unet_opt_zynq7020.hpp, the
// processes in sp_unet_direct_zynq7020_dataflow.hpp; here are its weights and the
// size of every process.
#include "sp_unet_small_opt_direct_zynq7020.hpp"

// The quantized weights, as the int8 integers q they are (value q * 2^-frac).
#include "../../../weights/quantized/sp_unet_small/sp_unet_small_int8.hpp"
namespace q = sp_unet_small_int8;

// The DSPs set the pace: every process takes at most T = 76 104 cycles a patch,
// about 649 ms for a 598 x 1092 capture at 78.1 MHz (the board's 12.8 ns). A conv
// layer takes (H x W + W + 1) x IN/PI x OUT/PO x 3/KR cycles a patch
// (conv3x3_grouped), with 3 x KR x PI x PO multipliers; the table is the
// fastest that fits 205 DSPs.
namespace cfg {
constexpr int P1 = 20;               // convsp1: 20 bands a chunk, 6 chunks
constexpr int SP1_PASSES = 12;       // ... x 12 passes over its pairs: 73 728 cycles a patch, 20 DSPs
constexpr bool ZSCORE_DSP = true;  // the z-score's 4 multipliers on DSPs
constexpr int SP2_PI = 4, SP2_PO = 1;  // convsp2 24 -> 12: 6 x 12 = 72 cycles a pixel (73 728), 4 DSPs
constexpr bool SP2_DSP = true;
constexpr int SP3_PI = 2, SP3_PO = 1;  // convsp3 12 -> 12: 6 x 12 = 72 cycles a pixel (73 728), 2 DSPs
constexpr bool SP3_DSP = true;
constexpr int FIN_PI = 1, FIN_PO = 1;  // final 8 -> 3: 8 x 3 = 24 cycles a pixel (24 576), 1 DSP
constexpr bool FIN_DSP = true;
//                       PI  PO KR DSP           channels  cycles a pixel          a patch    multipliers
typedef spu7_layer< 6, 1, 1, true> D1A;  // 12 -> 12   2 x 12 x 3 =   72 a pixel  x 1057 =  76 104   18
typedef spu7_layer< 6, 1, 1, true> D1B;  // 12 -> 12   2 x 12 x 3 =   72 a pixel  x 1057 =  76 104   18
typedef spu7_layer< 4, 1, 1, true> D2A;  // 12 -> 24   3 x 24 x 3 =  216 a pixel  x  273 =  58 968   12
typedef spu7_layer< 8, 1, 1, true> D2B;  // 24 -> 24   3 x 24 x 3 =  216 a pixel  x  273 =  58 968   24
typedef spu7_layer< 8, 1, 1, true> UP1;  // 24 -> 24   3 x 24 x 3 =  216 a pixel  x  273 =  58 968   24
typedef spu7_layer< 4, 1, 1, true> D3A;  // 24 -> 12   6 x 12 x 3 =  216 a pixel  x  273 =  58 968   12
typedef spu7_layer< 2, 1, 1, true> D3B;  // 12 -> 12   6 x 12 x 3 =  216 a pixel  x  273 =  58 968    6
typedef spu7_layer< 6, 1, 1, true> UP2;  // 12 -> 12   2 x 12 x 3 =   72 a pixel  x 1057 =  76 104   18
typedef spu7_layer< 8, 1, 1, true> D4A;  // 24 ->  8   3 x  8 x 3 =   72 a pixel  x 1057 =  76 104   24
typedef spu7_layer< 1, 1, 3, true> D4B;  //  8 ->  8   8 x  8 x 1 =   64 a pixel  x 1057 =  67 648    9
// DSPs: 165 convs + 20 convsp1 + 7 convsp2/3 and final + 4 z-score = 196 of 220 (the budget: 205).
}

#include "../sp_unet_small_opt_direct_zynq7020/sp_unet_direct_zynq7020_dataflow.hpp"

void sp_unet_small_opt_direct_zynq7020(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_out_t> dout,
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
