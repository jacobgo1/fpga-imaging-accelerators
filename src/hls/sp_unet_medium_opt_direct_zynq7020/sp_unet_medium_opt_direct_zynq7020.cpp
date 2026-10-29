#include "sp_unet_medium_opt_direct_zynq7020.hpp"

#include "../../../weights/quantized/sp_unet_medium/sp_unet_medium_int8.hpp"
namespace weights = sp_unet_medium_int8;

// Every process takes at most T = 228 528 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 4;                          // convsp1: 4 bands a chunk, 28 chunks
constexpr int CONVSP1_PASSES = 6;                                   // ... x 6 passes over its pairs: 172 032 cycles a patch, 8 DSPs
constexpr int SKIP_DEPTH = 1024;                                    // the skip FIFO, pixels (a patch: the block RAM is short)
constexpr bool ZSCORE_ON_DSPS = true;                               // the z-score's 4 multipliers on DSPs
constexpr int CONVSP2_IN_PER_CYCLE = 2, CONVSP2_OUT_PER_CYCLE = 1;  // convsp2 24 -> 16: 12 x 16 = 192 cycles a pixel (196 608), 2 DSPs
constexpr bool CONVSP2_ON_DSPS = true;
constexpr int CONVSP3_IN_PER_CYCLE = 1, CONVSP3_OUT_PER_CYCLE = 1;  // convsp3 16 -> 12: 16 x 12 = 192 cycles a pixel (196 608), 1 DSPs
constexpr bool CONVSP3_ON_DSPS = true;
constexpr int FINAL_IN_PER_CYCLE = 1, FINAL_OUT_PER_CYCLE = 1;      // final 12 -> 3: 12 x 3 = 36 cycles a pixel (36 864), 1 DSP
constexpr bool FINAL_ON_DSPS = true;
// The convs: channels, taps x in x out steps a pixel x pixels = cycles a patch, multipliers.
typedef lines_layer_size<12, 1, 1, true> DOUBLE1_CONV1;             // 12 -> 24  9 x  1 x 24 =  216 a pixel  x 1058 = 228 528   12
typedef lines_layer_size<24, 1, 1, true> DOUBLE1_CONV2;             // 24 -> 24  9 x  1 x 24 =  216 a pixel  x 1058 = 228 528   24
typedef lines_layer_size< 8, 2, 1, true> DOUBLE2_CONV1;             // 24 -> 48  9 x  3 x 24 =  648 a pixel  x  274 = 177 552   16
typedef lines_layer_size< 3, 3, 3, true> DOUBLE2_CONV2;             // 48 -> 48  3 x 16 x 16 =  768 a pixel  x  274 = 210 432   27
typedef lines_layer_size< 3, 3, 3, true> UP1_CONV;                  // 48 -> 48  3 x 16 x 16 =  768 a pixel  x  274 = 210 432   27
typedef lines_layer_size<16, 1, 1, true> DOUBLE3_CONV1;             // 48 -> 24  9 x  3 x 24 =  648 a pixel  x  274 = 177 552   16
typedef lines_layer_size< 8, 1, 1, true> DOUBLE3_CONV2;             // 24 -> 24  9 x  3 x 24 =  648 a pixel  x  274 = 177 552    8
typedef lines_layer_size<24, 1, 1, true> UP2_CONV;                  // 24 -> 24  9 x  1 x 24 =  216 a pixel  x 1058 = 228 528   24
typedef lines_layer_size<24, 1, 1, true> DOUBLE4_CONV1;             // 48 -> 12  9 x  2 x 12 =  216 a pixel  x 1058 = 228 528   24
typedef lines_layer_size< 6, 1, 1, true> DOUBLE4_CONV2;             // 12 -> 12  9 x  2 x 12 =  216 a pixel  x 1058 = 228 528    6
}

#include "../sp_unet_small_opt_direct_zynq7020/sp_unet_direct_zynq7020_dataflow.hpp"

void sp_unet_medium_opt_direct_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<score_word_t> dout,
                                        int height, int width, uint32_t& cycles)
{
    // depth is for co-simulation only.
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
