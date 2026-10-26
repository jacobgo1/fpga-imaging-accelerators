// sp_unet_large_opt_direct: the optimized SpecPW-UNet, large, with direct 3x3 convs
// (ZCU104): sp_unet_large_opt without Winograd, simpler and slower. The
// processes are in sp_unet_small_opt_direct/sp_unet_direct_dataflow.hpp; here
// are its weights and the size of every process.
#include "sp_unet_large_opt_direct.hpp"

// The quantized weights, as the int8 integers q they are (value q * 2^-frac).
#include "../../../weights/quantized/sp_unet_large/sp_unet_large_int8.hpp"
namespace q = sp_unet_large_int8;

// Every process takes at most T = 69 888 cycles a patch. A conv layer takes
// (H x W + W + 1) x IN/PI x OUT/PO cycles a patch (conv3x3_grouped), with
// 9 x PI x PO multipliers; the DSPs are kept at 1 200-1 500, for the clock.
namespace cfg {
constexpr int P1 = 8;               // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 128 DSPs
constexpr bool ZSCORE_DSP = false;  // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr int SP2_PO = 1;           // convsp2 32 -> 16: 16 cycles a pixel (16 384), 32 multipliers in LUTs
constexpr int SP3_PO = 1;           // convsp3 16 -> 16: 16 (16 384), 16 in LUTs
constexpr int FIN_PO = 1;           // final 16 -> 3: 3 (3 072), 16 in LUTs
//                       PI  PO DSP          channels  cycles a pixel  a patch (+ W + 1)   multipliers
typedef spu_layer< 8, 1, true> D1A;  // 16 -> 32   2 x 32 = 64    x 1 057 = 67 648      72
typedef spu_layer<16, 1, true> D1B;  // 32 -> 32   2 x 32 = 64    x 1 057 = 67 648     144
typedef spu_layer< 8, 1, true> D2A;  // 32 -> 64   4 x 64 = 256   x 273 = 69 888        72
typedef spu_layer<16, 1, true> D2B;  // 64 -> 64   4 x 64 = 256   x 273 = 69 888       144
typedef spu_layer<16, 1, true> UP1;  // 64 -> 64   4 x 64 = 256   x 273 = 69 888       144
typedef spu_layer< 8, 1, true> D3A;  // 64 -> 32   8 x 32 = 256   x 273 = 69 888        72
typedef spu_layer< 4, 1, true> D3B;  // 32 -> 32   8 x 32 = 256   x 273 = 69 888        36
typedef spu_layer<16, 1, true> UP2;  // 32 -> 32   2 x 32 = 64    x 1 057 = 67 648     144
typedef spu_layer<16, 1, true> D4A;  // 64 -> 16   4 x 16 = 64    x 1 057 = 67 648     144
typedef spu_layer< 4, 1, true> D4B;  // 16 -> 16   4 x 16 = 64    x 1 057 = 67 648      36
// DSPs: 1 008 direct + 128 convsp1 = 1 136 of 1 728 (66 %); the next table, 33 824 cycles, needs ~2 100.
}

#include "../sp_unet_small_opt_direct/sp_unet_direct_dataflow.hpp"

void sp_unet_large_opt_direct(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_raw_t> din1,
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
