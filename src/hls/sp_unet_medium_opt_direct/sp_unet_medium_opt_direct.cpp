#include "sp_unet_medium_opt_direct.hpp"

#include "../../../weights/quantized/sp_unet_medium/sp_unet_medium_int8.hpp"
namespace weights = sp_unet_medium_int8;

// Every process takes at most T = 34 944 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 8;      // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 96 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr bool CONVSP2_3_ON_DSPS = true;        // convsp2 and convsp3 on DSPs: in LUTs they took ~25 k LUTs (small)
constexpr int CONVSP2_OUT_PER_CYCLE = 1;        // convsp2 24 -> 16: 16 cycles a pixel (16 384), 24 DSPs
constexpr int CONVSP3_OUT_PER_CYCLE = 1;        // convsp3 16 -> 12: 12 (12 288), 16 DSPs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 12 -> 3: 3 (3 072), 12 in LUTs
// The convs: channels, cycles a pixel x pixels (+ a row) = cycles a patch, multipliers.
typedef layer_size< 3, 3, true> DOUBLE1_CONV1;  // 12 -> 24   4 x 8 = 32     x 1 057 = 33 824      81
typedef layer_size< 6, 3, true> DOUBLE1_CONV2;  // 24 -> 24   4 x 8 = 32     x 1 057 = 33 824     162
typedef layer_size< 3, 3, true> DOUBLE2_CONV1;  // 24 -> 48   8 x 16 = 128   x 273 = 34 944        81
typedef layer_size< 6, 3, true> DOUBLE2_CONV2;  // 48 -> 48   8 x 16 = 128   x 273 = 34 944       162
typedef layer_size< 6, 3, true> UP1_CONV;       // 48 -> 48   8 x 16 = 128   x 273 = 34 944       162
typedef layer_size< 3, 3, true> DOUBLE3_CONV1;  // 48 -> 24   16 x 8 = 128   x 273 = 34 944        81
typedef layer_size< 6, 1, true> DOUBLE3_CONV2;  // 24 -> 24   4 x 24 = 96    x 273 = 26 208        54
typedef layer_size< 6, 3, true> UP2_CONV;       // 24 -> 24   4 x 8 = 32     x 1 057 = 33 824     162
typedef layer_size< 6, 3, true> DOUBLE4_CONV1;  // 48 -> 12   8 x 4 = 32     x 1 057 = 33 824     162
typedef layer_size< 6, 1, true> DOUBLE4_CONV2;  // 12 -> 12   2 x 12 = 24    x 1 057 = 25 368      54
}

#include "../sp_unet_small_opt_direct/sp_unet_direct_dataflow.hpp"

void sp_unet_medium_opt_direct(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
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
    sp_unet_direct_dataflow(din0, din1, din2, din3, dout, height, width, cycles);
}
