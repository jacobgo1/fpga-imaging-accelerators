// sp_unet_large_opt: the optimized SpecPW-UNet, large, on a whole raw capture (ZCU104).
// The design is in sp_unet_small_opt/sp_unet_opt.hpp, the processes in
// sp_unet_opt_dataflow.hpp; here are its weights and the size of every process.
#include "sp_unet_large_opt.hpp"

// The quantized weights, as the int8 integers q they are (value q * 2^-frac):
// this kernel multiplies integers. The scales become shifts.
#include "../../../weights/quantized/sp_unet_large/sp_unet_large_int8.hpp"
namespace q = sp_unet_large_int8;

// Every process takes about T = 16 384 cycles a patch: none waits for another.
// A Winograd layer takes IN/PI x OUT/PO cycles a block (or, an UpBlock, a
// low-res pixel), with 16 x PI x PO multipliers (9 for an UpBlock).
namespace cfg {
constexpr int P1 = 8;               // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 128 DSPs
constexpr bool ZSCORE_DSP = false;  // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr int SP2_PO = 1;           // convsp2 32 -> 16: 16 cycles a pixel (16 384), 32 multipliers in LUTs
constexpr int SP3_PO = 1;           // convsp3 16 -> 16: 16 (16 384), 16 in LUTs
constexpr int FIN_PO = 1;           // final 16 -> 3: 3 (3 072), 16 in LUTs
//                      PI PO DSP             channels   cycles a block   a patch    multipliers
typedef spu_layer<2, 4, false> D1A; // 16 -> 32      8 x 8 = 64  x 256 = 16 384      128 in LUTs
typedef spu_layer<4, 4, true>  D1B; // 32 -> 32      8 x 8 = 64  x 256 = 16 384      256
typedef spu_layer<2, 4, true>  D2A; // 32 -> 64    16 x 16 = 256 x  64 = 16 384      128
typedef spu_layer<4, 4, true>  D2B; // 64 -> 64    16 x 16 = 256 x  64 = 16 384      256
typedef spu_layer<4, 4, true>  UP1; // 64 -> 64    16 x 16 = 256 x  64 = 16 384      144 (9 x PI x PO)
typedef spu_layer<4, 2, true>  D3A; // 64 -> 32    16 x 16 = 256 x  64 = 16 384      128
typedef spu_layer<2, 2, false> D3B; // 32 -> 32    16 x 16 = 256 x  64 = 16 384       64 in LUTs
typedef spu_layer<4, 4, true>  UP2; // 32 -> 32      8 x 8 = 64  x 256 = 16 384      144
typedef spu_layer<4, 4, true>  D4A; // 64 -> 16     16 x 4 = 64  x 256 = 16 384      256
typedef spu_layer<2, 2, false> D4B; // 16 -> 16      8 x 8 = 64  x 256 = 16 384       64 in LUTs
// DSPs: 1 312 Winograd + 128 convsp1 = 1 440 of 1 728 (83 %); 256 Winograd multipliers in LUTs.
}

#include "../sp_unet_small_opt/sp_unet_opt_dataflow.hpp"

void sp_unet_large_opt(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_raw_t> din1,
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
