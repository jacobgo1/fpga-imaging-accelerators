#include "justoliunet_bn_simple.hpp"

#include "../../../../weights/quantized/justoliunet_bn/justoliunet_bn_int8.hpp"
#include "../../1D-Justo-LiuNet/justoliunet_opt/justoliunet_opt_formats.hpp"
namespace weights = justoliunet_bn_int8;
namespace fmt = justoliunet_bn_formats;
#include "../../1D-Justo-LiuNet/justoliunet_simple/justoliunet_simple_chain.hpp"

void justoliunet_bn_simple(const cube_word_t* din0, score_word_t* dout, int height, int width)
{
    // depth is for co-simulation only: the testbench's 9 x 7 image.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=945 max_read_burst_length=256
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem1 offset=slave depth=63
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    justoliunet_simple_chain(din0, dout, height, width);
}
