// sp_unet_medium_opt: the optimized SpecPW-UNet, medium, on a whole raw capture (ZCU104).
// The design is in sp_unet_small_opt/sp_unet_opt.hpp, the processes in
// sp_unet_opt_dataflow.hpp; here are its weights and the size of every process.
#include "sp_unet_medium_opt.hpp"

// The quantized weights, as the int8 integers q they are (value q * 2^-frac):
// this kernel multiplies integers. The scales become shifts.
#include "../../../weights/quantized/sp_unet_medium/sp_unet_medium_int8.hpp"
namespace q = sp_unet_medium_int8;

// Every process takes about T = 12 288 cycles a patch: none waits for another.
// A Winograd layer takes IN/PI x OUT/PO cycles a block (or, an UpBlock, a
// low-res pixel), with 16 x PI x PO multipliers (9 for an UpBlock). T = 9 216
// fits in time but needs ~1 760 DSPs; 12 288 keeps them below 1 500, for the clock.
namespace cfg {
constexpr int P1 = 16;              // convsp1: 16 bands a cycle, 7 cycles a pixel (7 168), 192 DSPs
constexpr bool ZSCORE_DSP = false;  // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr int SP2_PO = 2;           // convsp2 24 -> 16: 8 cycles a pixel (8 192), 48 multipliers in LUTs
constexpr int SP3_PO = 1;           // convsp3 16 -> 12: 12 (12 288), 16 in LUTs
constexpr int FIN_PO = 1;           // final 12 -> 3: 3 (3 072), 12 in LUTs
//                      PI PO DSP            channels   cycles a block   a patch    multipliers
typedef spu_layer<2, 3, true> D1A;  // 12 -> 24      6 x 8 = 48  x 256 = 12 288       96
typedef spu_layer<4, 3, true> D1B;  // 24 -> 24      6 x 8 = 48  x 256 = 12 288      192
typedef spu_layer<2, 3, true> D2A;  // 24 -> 48    12 x 16 = 192 x  64 = 12 288       96
typedef spu_layer<4, 3, true> D2B;  // 48 -> 48    12 x 16 = 192 x  64 = 12 288      192
typedef spu_layer<4, 3, true> UP1;  // 48 -> 48    12 x 16 = 192 x  64 = 12 288      108 (9 x PI x PO)
typedef spu_layer<3, 2, true> D3A;  // 48 -> 24    16 x 12 = 192 x  64 = 12 288       96
typedef spu_layer<3, 1, true> D3B;  // 24 -> 24     8 x 24 = 192 x  64 = 12 288       48
typedef spu_layer<4, 3, true> UP2;  // 24 -> 24      6 x 8 = 48  x 256 = 12 288      108
typedef spu_layer<4, 3, true> D4A;  // 48 -> 12     12 x 4 = 48  x 256 = 12 288      192
typedef spu_layer<3, 1, true> D4B;  // 12 -> 12     4 x 12 = 48  x 256 = 12 288       48
// DSPs: 1 176 Winograd + 192 convsp1 = 1 368 of 1 728 (79 %); no Winograd multiplier in LUTs.
}

#include "../sp_unet_small_opt/sp_unet_opt_dataflow.hpp"

void sp_unet_medium_opt(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_raw_t> din1,
                        hls::burst_maxi<spu_raw_t> din2, hls::burst_maxi<spu_raw_t> din3,
                        hls::burst_maxi<spu_out_t> dout, int height, int width, uint32_t& cycles)
{
// Five AXI masters, each on its own port to DDR (boards/zcu104/system.tcl):
// four read the cube in parallel (point all of them at it) through HP0-3,
// one writes the scores through HPC0.
// depth is only for co-simulation: the testbench's 40 x 50 image.
#pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
#pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
#pragma HLS INTERFACE mode=m_axi port=din2 bundle=gmem2 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
#pragma HLS INTERFACE mode=m_axi port=din3 bundle=gmem3 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
#pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem4 offset=slave depth=2000 max_write_burst_length=32 num_write_outstanding=16
#pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=din2   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=din3   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=height bundle=control
#pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
#pragma HLS INTERFACE mode=s_axilite port=cycles bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    sp_unet_opt_dataflow(din0, din1, din2, din3, dout, height, width, cycles);
}
