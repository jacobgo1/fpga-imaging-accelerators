#include "sp_unet_small_opt.hpp"

#include "../../../weights/quantized/sp_unet_small/sp_unet_small_int8.hpp"
namespace weights = sp_unet_small_int8;

// Every process takes at most T = 4 096 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 32;    // convsp1: 32 bands a cycle, 4 cycles a pixel (4 096), 384 DSPs
constexpr bool ZSCORE_ON_DSPS = false;         // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr int CONVSP2_OUT_PER_CYCLE = 3;       // convsp2 24 -> 12: 4 cycles a pixel (4 096), 72 multipliers in LUTs
constexpr int CONVSP3_OUT_PER_CYCLE = 3;       // convsp3 12 -> 12: 4 (4 096), 36 in LUTs
constexpr int FINAL_OUT_PER_CYCLE = 1;         // final 8 -> 3: 3 (3 072), 8 in LUTs
// The Winograd convs: channels, cycles a block x blocks = cycles a patch, multipliers.
typedef layer_size<3, 3, true> DOUBLE1_CONV1;  // 12 -> 12     4 x 4 = 16   x 256 = 4 096      144
typedef layer_size<3, 3, true> DOUBLE1_CONV2;  // 12 -> 12     4 x 4 = 16   x 256 = 4 096      144
typedef layer_size<2, 3, true> DOUBLE2_CONV1;  // 12 -> 24     6 x 8 = 48   x  64 = 3 072       96
typedef layer_size<3, 3, true> DOUBLE2_CONV2;  // 24 -> 24     8 x 8 = 64   x  64 = 4 096      144
typedef layer_size<3, 3, true> UP1_CONV;       // 24 -> 24     8 x 8 = 64   x  64 = 4 096       81 (9 x PI x PO)
typedef layer_size<3, 2, true> DOUBLE3_CONV1;  // 24 -> 12     8 x 6 = 48   x  64 = 3 072       96
typedef layer_size<3, 1, true> DOUBLE3_CONV2;  // 12 -> 12    4 x 12 = 48   x  64 = 3 072       48
typedef layer_size<3, 3, true> UP2_CONV;       // 12 -> 12     4 x 4 = 16   x 256 = 4 096       81
typedef layer_size<6, 2, true> DOUBLE4_CONV1;  // 24 -> 8      4 x 4 = 16   x 256 = 4 096      192
typedef layer_size<2, 2, true> DOUBLE4_CONV2;  // 8 -> 8      4 x 4 = 16   x 256 = 4 096       64
}

#include "../sp_unet_small_opt/sp_unet_opt_dataflow.hpp"

void sp_unet_small_opt(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
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
