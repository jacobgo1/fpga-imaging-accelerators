#include "sp_unet_large_opt.hpp"

#include "../../../weights/quantized/sp_unet_large/sp_unet_large_int8.hpp"
namespace weights = sp_unet_large_int8;

// Every process takes at most T = 16 384 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 8;      // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 128 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr int CONVSP2_OUT_PER_CYCLE = 1;        // convsp2 32 -> 16: 16 cycles a pixel (16 384), 32 multipliers in LUTs
constexpr int CONVSP3_OUT_PER_CYCLE = 1;        // convsp3 16 -> 16: 16 (16 384), 16 in LUTs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 16 -> 3: 3 (3 072), 16 in LUTs
// The Winograd convs: channels, cycles a block x blocks = cycles a patch, multipliers.
typedef layer_size<2, 4, false> DOUBLE1_CONV1;  // 16 -> 32      8 x 8 = 64  x 256 = 16 384      128 in LUTs
typedef layer_size<4, 4, true>  DOUBLE1_CONV2;  // 32 -> 32      8 x 8 = 64  x 256 = 16 384      256
typedef layer_size<2, 4, true>  DOUBLE2_CONV1;  // 32 -> 64    16 x 16 = 256 x  64 = 16 384      128
typedef layer_size<4, 4, true>  DOUBLE2_CONV2;  // 64 -> 64    16 x 16 = 256 x  64 = 16 384      256
typedef layer_size<4, 4, true>  UP1_CONV;       // 64 -> 64    16 x 16 = 256 x  64 = 16 384      144 (9 x PI x PO)
typedef layer_size<4, 2, true>  DOUBLE3_CONV1;  // 64 -> 32    16 x 16 = 256 x  64 = 16 384      128
typedef layer_size<2, 2, false> DOUBLE3_CONV2;  // 32 -> 32    16 x 16 = 256 x  64 = 16 384       64 in LUTs
typedef layer_size<4, 4, true>  UP2_CONV;       // 32 -> 32      8 x 8 = 64  x 256 = 16 384      144
typedef layer_size<4, 4, true>  DOUBLE4_CONV1;  // 64 -> 16     16 x 4 = 64  x 256 = 16 384      256
typedef layer_size<2, 2, false> DOUBLE4_CONV2;  // 16 -> 16      8 x 8 = 64  x 256 = 16 384       64 in LUTs
}

#include "../sp_unet_small_opt/sp_unet_opt_dataflow.hpp"

void sp_unet_large_opt(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                       hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                       hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles)
{
    // depth is for co-simulation only.
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
