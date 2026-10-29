#include "sp_unet_small_opt_direct.hpp"

#include "../../../weights/quantized/sp_unet_small/sp_unet_small_int8.hpp"
namespace weights = sp_unet_small_int8;

// Every process takes at most T = 9 828 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 16;     // convsp1: 16 bands a cycle, 7 cycles a pixel (7 168), 192 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr bool CONVSP2_3_ON_DSPS = true;        // convsp2 and convsp3 on DSPs: in LUTs they took ~25 k LUTs (small)
constexpr int CONVSP2_OUT_PER_CYCLE = 2;        // convsp2 24 -> 12: 6 cycles a pixel (6 144), 48 DSPs
constexpr int CONVSP3_OUT_PER_CYCLE = 2;        // convsp3 12 -> 12: 6 (6 144), 24 DSPs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 8 -> 3: 3 (3 072), 8 in LUTs
// The convs: channels, cycles a pixel x pixels (+ a row) = cycles a patch, multipliers.
typedef layer_size< 4, 4, true> DOUBLE1_CONV1;  // 12 -> 12   3 x 3 = 9      x 1 057 = 9 513      144
typedef layer_size< 4, 4, true> DOUBLE1_CONV2;  // 12 -> 12   3 x 3 = 9      x 1 057 = 9 513      144
typedef layer_size< 4, 2, true> DOUBLE2_CONV1;  // 12 -> 24   3 x 12 = 36    x 273 = 9 828         72
typedef layer_size< 8, 2, true> DOUBLE2_CONV2;  // 24 -> 24   3 x 12 = 36    x 273 = 9 828        144
typedef layer_size< 8, 2, true> UP1_CONV;       // 24 -> 24   3 x 12 = 36    x 273 = 9 828        144
typedef layer_size< 8, 1, true> DOUBLE3_CONV1;  // 24 -> 12   3 x 12 = 36    x 273 = 9 828         72
typedef layer_size< 4, 1, true> DOUBLE3_CONV2;  // 12 -> 12   3 x 12 = 36    x 273 = 9 828         36
typedef layer_size< 4, 4, true> UP2_CONV;       // 12 -> 12   3 x 3 = 9      x 1 057 = 9 513      144
typedef layer_size<24, 1, true> DOUBLE4_CONV1;  // 24 -> 8    1 x 8 = 8      x 1 057 = 8 456      216
typedef layer_size< 8, 1, true> DOUBLE4_CONV2;  // 8 -> 8     1 x 8 = 8      x 1 057 = 8 456       72
}

#include "../sp_unet_small_opt_direct/sp_unet_direct_dataflow.hpp"

void sp_unet_small_opt_direct(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
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
