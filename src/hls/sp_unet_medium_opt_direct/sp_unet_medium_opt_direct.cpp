// sp_unet_medium_opt_direct: the optimized SpecPW-UNet, medium, with direct 3x3 convs
// (ZCU104): sp_unet_medium_opt without Winograd, simpler and slower. The
// processes are in sp_unet_small_opt_direct/sp_unet_direct_dataflow.hpp; here
// are its weights and the size of every process.
#include "sp_unet_medium_opt_direct.hpp"

// The quantized weights, as the int8 integers q they are (value q * 2^-frac).
#include "../../../weights/quantized/sp_unet_medium/sp_unet_medium_int8.hpp"
namespace q = sp_unet_medium_int8;

// Every process takes at most T = 34 944 cycles a patch. A conv layer takes
// (H x W + W + 1) x IN/PI x OUT/PO cycles a patch (conv3x3_grouped), with
// 9 x PI x PO multipliers; the DSPs are kept at 1 200-1 500, for the clock.
namespace cfg {
constexpr int P1 = 8;               // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 96 DSPs
constexpr bool ZSCORE_DSP = false;  // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr int SP2_PO = 1;           // convsp2 24 -> 16: 16 cycles a pixel (16 384), 24 multipliers in LUTs
constexpr int SP3_PO = 1;           // convsp3 16 -> 12: 12 (12 288), 16 in LUTs
constexpr int FIN_PO = 1;           // final 12 -> 3: 3 (3 072), 12 in LUTs
//                       PI  PO DSP          channels  cycles a pixel  a patch (+ W + 1)   multipliers
typedef spu_layer< 3, 3, true> D1A;  // 12 -> 24   4 x 8 = 32     x 1 057 = 33 824      81
typedef spu_layer< 6, 3, true> D1B;  // 24 -> 24   4 x 8 = 32     x 1 057 = 33 824     162
typedef spu_layer< 3, 3, true> D2A;  // 24 -> 48   8 x 16 = 128   x 273 = 34 944        81
typedef spu_layer< 6, 3, true> D2B;  // 48 -> 48   8 x 16 = 128   x 273 = 34 944       162
typedef spu_layer< 6, 3, true> UP1;  // 48 -> 48   8 x 16 = 128   x 273 = 34 944       162
typedef spu_layer< 3, 3, true> D3A;  // 48 -> 24   16 x 8 = 128   x 273 = 34 944        81
typedef spu_layer< 6, 1, true> D3B;  // 24 -> 24   4 x 24 = 96    x 273 = 26 208        54
typedef spu_layer< 6, 3, true> UP2;  // 24 -> 24   4 x 8 = 32     x 1 057 = 33 824     162
typedef spu_layer< 6, 3, true> D4A;  // 48 -> 12   8 x 4 = 32     x 1 057 = 33 824     162
typedef spu_layer< 6, 1, true> D4B;  // 12 -> 12   2 x 12 = 24    x 1 057 = 25 368      54
// DSPs: 1 161 direct + 96 convsp1 = 1 257 of 1 728 (73 %).
}

#include "../sp_unet_small_opt_direct/sp_unet_direct_dataflow.hpp"

void sp_unet_medium_opt_direct(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_raw_t> din1,
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
    sp_unet_direct_dataflow(din0, din1, din2, din3, dout, height, width, cycles);
}
