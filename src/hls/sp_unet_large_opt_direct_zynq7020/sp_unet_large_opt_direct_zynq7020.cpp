#include "sp_unet_large_opt_direct_zynq7020.hpp"

#include "../../../weights/quantized/sp_unet_large/sp_unet_large_int8.hpp"
namespace weights = sp_unet_large_int8;

// Every process takes at most T = 420 864 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 4;                          // convsp1: 4 bands a chunk, 28 chunks
constexpr int CONVSP1_PASSES = 8;                                   // ... x 8 passes over its pairs: 229 376 cycles a patch, 8 DSPs
constexpr int SKIP_DEPTH = 1024;                                    // the skip FIFO, pixels (a patch: the block RAM is short)
constexpr bool ZSCORE_ON_DSPS = true;                               // the z-score's 4 multipliers on DSPs
constexpr int CONVSP2_IN_PER_CYCLE = 2, CONVSP2_OUT_PER_CYCLE = 1;  // convsp2 32 -> 16: 16 x 16 = 256 cycles a pixel (262 144), 2 DSPs
constexpr bool CONVSP2_ON_DSPS = true;
constexpr int CONVSP3_IN_PER_CYCLE = 1, CONVSP3_OUT_PER_CYCLE = 1;  // convsp3 16 -> 16: 16 x 16 = 256 cycles a pixel (262 144), 1 DSPs
constexpr bool CONVSP3_ON_DSPS = true;
constexpr int FINAL_IN_PER_CYCLE = 1, FINAL_OUT_PER_CYCLE = 1;      // final 16 -> 3: 16 x 3 = 48 cycles a pixel (49 152), 1 DSP
constexpr bool FINAL_ON_DSPS = true;
// The convs: channels, taps x in x out steps a pixel x pixels = cycles a patch, multipliers.
typedef lines_layer_size< 4, 1, 3, true> DOUBLE1_CONV1;             // 16 -> 32  3 x  4 x 32 =  384 a pixel  x 1058 = 406 272   12
typedef lines_layer_size< 8, 1, 3, true> DOUBLE1_CONV2;             // 32 -> 32  3 x  4 x 32 =  384 a pixel  x 1058 = 406 272   24
typedef lines_layer_size< 4, 1, 3, true> DOUBLE2_CONV1;             // 32 -> 64  3 x  8 x 64 = 1536 a pixel  x  274 = 420 864   12
typedef lines_layer_size< 8, 1, 3, true> DOUBLE2_CONV2;             // 64 -> 64  3 x  8 x 64 = 1536 a pixel  x  274 = 420 864   24
typedef lines_layer_size< 8, 1, 3, true> UP1_CONV;                  // 64 -> 64  3 x  8 x 64 = 1536 a pixel  x  274 = 420 864   24
typedef lines_layer_size< 4, 1, 3, true> DOUBLE3_CONV1;             // 64 -> 32  3 x 16 x 32 = 1536 a pixel  x  274 = 420 864   12
typedef lines_layer_size< 2, 1, 3, true> DOUBLE3_CONV2;             // 32 -> 32  3 x 16 x 32 = 1536 a pixel  x  274 = 420 864    6
typedef lines_layer_size< 8, 1, 3, true> UP2_CONV;                  // 32 -> 32  3 x  4 x 32 =  384 a pixel  x 1058 = 406 272   24
typedef lines_layer_size< 8, 1, 3, true> DOUBLE4_CONV1;             // 64 -> 16  3 x  8 x 16 =  384 a pixel  x 1058 = 406 272   24
typedef lines_layer_size< 2, 1, 3, true> DOUBLE4_CONV2;             // 16 -> 16  3 x  8 x 16 =  384 a pixel  x 1058 = 406 272    6
}

#include "../sp_unet_small_opt_direct_zynq7020/sp_unet_direct_zynq7020_dataflow.hpp"

void sp_unet_large_opt_direct_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<score_word_t> dout,
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
