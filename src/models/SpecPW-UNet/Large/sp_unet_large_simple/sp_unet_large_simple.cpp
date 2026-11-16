#include "sp_unet_large_simple.hpp"

#include "../../../../../weights/quantized/sp_unet_large/sp_unet_large_int8.hpp"
namespace weights = sp_unet_large_int8;

// Each conv's inputs (UI) and outputs (UO) a cycle; cycles = H * W * K*K * IN/UI * OUT/UO.
// T = 73728 cycles a patch (~196 ms an image at 250 MHz), 964 multipliers.
namespace sizes {
//                                           in -> out  size     cycles  multipliers
typedef layer_size<55, 1> CONVSP1;         // 110 -> 32  32 x 32   65536    55
typedef layer_size< 8, 1> CONVSP2;         //  32 -> 16  32 x 32   65536     8
typedef layer_size< 4, 1> CONVSP3;         //  16 -> 16  32 x 32   65536     4
typedef layer_size<16, 4> DOUBLE1_CONV1;   //  16 -> 32  32 x 32   73728    64
typedef layer_size<32, 4> DOUBLE1_CONV2;   //  32 -> 32  32 x 32   73728   128
typedef layer_size<32, 2> DOUBLE2_CONV1;   //  32 -> 64  16 x 16   73728    64
typedef layer_size<64, 2> DOUBLE2_CONV2;   //  64 -> 64  16 x 16   73728   128
typedef layer_size<64, 2> UP1_CONV;        //  64 -> 64  16 x 16   73728   128
typedef layer_size<64, 1> DOUBLE3_CONV1;   //  64 -> 32  16 x 16   73728    64
typedef layer_size<32, 1> DOUBLE3_CONV2;   //  32 -> 32  16 x 16   73728    32
typedef layer_size<32, 4> UP2_CONV;        //  32 -> 32  32 x 32   73728   128
typedef layer_size<32, 4> DOUBLE4_CONV1;   //  64 -> 16  32 x 32   73728   128
typedef layer_size<16, 2> DOUBLE4_CONV2;   //  16 -> 16  32 x 32   73728    32
typedef layer_size< 1, 1> FINAL_CONV;      //  16 ->  3  32 x 32   49152     1
}
#include "../../Small/sp_unet_small_simple/sp_unet_simple_chain.hpp"

void sp_unet_large_simple(const cube_word_t* din0, score_word_t* dout, int height, int width)
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
