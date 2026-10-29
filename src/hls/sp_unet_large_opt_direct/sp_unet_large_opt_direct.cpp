#include "sp_unet_large_opt_direct.hpp"

#include "../../../weights/quantized/sp_unet_large/sp_unet_large_int8.hpp"
namespace weights = sp_unet_large_int8;

// Every process takes at most T = 69 888 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 8;      // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 128 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr bool CONVSP2_3_ON_DSPS = true;        // convsp2 and convsp3 on DSPs: in LUTs they took ~25 k LUTs (small)
constexpr int CONVSP2_OUT_PER_CYCLE = 1;        // convsp2 32 -> 16: 16 cycles a pixel (16 384), 32 DSPs
constexpr int CONVSP3_OUT_PER_CYCLE = 1;        // convsp3 16 -> 16: 16 (16 384), 16 DSPs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 16 -> 3: 3 (3 072), 16 DSPs
// The convs: channels, cycles a pixel x pixels (+ a row) = cycles a patch, multipliers.
typedef layer_size< 8, 1, true> DOUBLE1_CONV1;  // 16 -> 32   2 x 32 = 64    x 1 057 = 67 648      72
typedef layer_size<16, 1, true> DOUBLE1_CONV2;  // 32 -> 32   2 x 32 = 64    x 1 057 = 67 648     144
typedef layer_size< 8, 1, true> DOUBLE2_CONV1;  // 32 -> 64   4 x 64 = 256   x 273 = 69 888        72
typedef layer_size<16, 1, true> DOUBLE2_CONV2;  // 64 -> 64   4 x 64 = 256   x 273 = 69 888       144
typedef layer_size<16, 1, true> UP1_CONV;       // 64 -> 64   4 x 64 = 256   x 273 = 69 888       144
typedef layer_size< 8, 1, true> DOUBLE3_CONV1;  // 64 -> 32   8 x 32 = 256   x 273 = 69 888        72
typedef layer_size< 4, 1, true> DOUBLE3_CONV2;  // 32 -> 32   8 x 32 = 256   x 273 = 69 888        36
typedef layer_size<16, 1, true> UP2_CONV;       // 32 -> 32   2 x 32 = 64    x 1 057 = 67 648     144
typedef layer_size<16, 1, true> DOUBLE4_CONV1;  // 64 -> 16   4 x 16 = 64    x 1 057 = 67 648     144
typedef layer_size< 4, 1, true> DOUBLE4_CONV2;  // 16 -> 16   4 x 16 = 64    x 1 057 = 67 648      36
}

#include "../sp_unet_small_opt_direct/sp_unet_direct_dataflow.hpp"

void sp_unet_large_opt_direct(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
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
