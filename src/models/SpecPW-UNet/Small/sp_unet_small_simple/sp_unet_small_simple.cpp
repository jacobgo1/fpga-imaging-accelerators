#include "sp_unet_small_simple.hpp"

#include "../../../../../weights/quantized/sp_unet_small/sp_unet_small_int8.hpp"
namespace weights = sp_unet_small_int8;

// Each conv's inputs (UI) and outputs (UO) a cycle; cycles = H * W * K*K * IN/UI * OUT/UO.
// T = 18432 cycles a patch (~49 ms an image at 250 MHz), 769 multipliers.
namespace sizes {
//                                           in -> out  size     cycles  multipliers
typedef layer_size<55, 3> CONVSP1;         // 110 -> 24  32 x 32   16384   165
typedef layer_size< 8, 2> CONVSP2;         //  24 -> 12  32 x 32   18432    16
typedef layer_size< 4, 2> CONVSP3;         //  12 -> 12  32 x 32   18432     8
typedef layer_size<12, 6> DOUBLE1_CONV1;   //  12 -> 12  32 x 32   18432    72
typedef layer_size<12, 6> DOUBLE1_CONV2;   //  12 -> 12  32 x 32   18432    72
typedef layer_size<12, 3> DOUBLE2_CONV1;   //  12 -> 24  16 x 16   18432    36
typedef layer_size<24, 3> DOUBLE2_CONV2;   //  24 -> 24  16 x 16   18432    72
typedef layer_size<24, 3> UP1_CONV;        //  24 -> 24  16 x 16   18432    72
typedef layer_size<12, 3> DOUBLE3_CONV1;   //  24 -> 12  16 x 16   18432    36
typedef layer_size< 6, 3> DOUBLE3_CONV2;   //  12 -> 12  16 x 16   18432    18
typedef layer_size<12, 6> UP2_CONV;        //  12 -> 12  32 x 32   18432    72
typedef layer_size<12, 8> DOUBLE4_CONV1;   //  24 ->  8  32 x 32   18432    96
typedef layer_size< 8, 4> DOUBLE4_CONV2;   //   8 ->  8  32 x 32   18432    32
typedef layer_size< 2, 1> FINAL_CONV;      //   8 ->  3  32 x 32   12288     2
}
#include "sp_unet_simple_chain.hpp"

void sp_unet_small_simple(const cube_word_t* din0, score_word_t* dout, int height, int width)
{
    // depth is for co-simulation only: the testbench's 40 x 50 image.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=30000 max_read_burst_length=256
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem1 offset=slave depth=2000
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    sp_unet_simple(din0, dout, height, width);
}
