#include "sp_unet_medium_simple.hpp"

#include "../../../../../weights/quantized/sp_unet_medium/sp_unet_medium_int8.hpp"
namespace weights = sp_unet_medium_int8;

// Each conv's inputs (UI) and outputs (UO) a cycle; cycles = H * W * K*K * IN/UI * OUT/UO.
// T = 41472 cycles a patch (~110 ms an image at 250 MHz), 1041 multipliers.
namespace sizes {
//                                           in -> out  size     cycles  multipliers
typedef layer_size<22, 3> CONVSP1;         // 110 -> 24  32 x 32   40960    66
typedef layer_size<12, 1> CONVSP2;         //  24 -> 16  32 x 32   32768    12
typedef layer_size< 2, 3> CONVSP3;         //  16 -> 12  32 x 32   32768     6
typedef layer_size<12, 6> DOUBLE1_CONV1;   //  12 -> 24  32 x 32   36864    72
typedef layer_size<24, 6> DOUBLE1_CONV2;   //  24 -> 24  32 x 32   36864   144
typedef layer_size< 8, 8> DOUBLE2_CONV1;   //  24 -> 48  16 x 16   41472    64
typedef layer_size<16, 8> DOUBLE2_CONV2;   //  48 -> 48  16 x 16   41472   128
typedef layer_size<16, 8> UP1_CONV;        //  48 -> 48  16 x 16   41472   128
typedef layer_size<16, 4> DOUBLE3_CONV1;   //  48 -> 24  16 x 16   41472    64
typedef layer_size< 8, 4> DOUBLE3_CONV2;   //  24 -> 24  16 x 16   41472    32
typedef layer_size<24, 6> UP2_CONV;        //  24 -> 24  32 x 32   36864   144
typedef layer_size<24, 6> DOUBLE4_CONV1;   //  48 -> 12  32 x 32   36864   144
typedef layer_size<12, 3> DOUBLE4_CONV2;   //  12 -> 12  32 x 32   36864    36
typedef layer_size< 1, 1> FINAL_CONV;      //  12 ->  3  32 x 32   36864     1
}
#include "../../Small/sp_unet_small_simple/sp_unet_simple_chain.hpp"

void sp_unet_medium_simple(const cube_word_t* din0, score_word_t* dout, int height, int width)
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
