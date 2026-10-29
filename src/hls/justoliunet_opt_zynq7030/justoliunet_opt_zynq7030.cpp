#include "../../../weights/quantized/justoliunet/justoliunet_int8.hpp"
#include "../justoliunet_opt/justoliunet_opt_formats.hpp"
namespace weights = justoliunet_int8;
namespace model_formats = justoliunet_formats;
#include "justoliunet_opt_zynq7030_dataflow.hpp"

void justoliunet_opt_zynq7030(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<score_word_t> dout,
                              int height, int width, uint32_t& cycles)
{
    // The reader on HP0, the writer on HP1. depth is for co-simulation only.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=1890 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem1 offset=slave depth=63 max_write_burst_length=64 num_write_outstanding=16
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=cycles bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    justoliunet_opt_zynq7030_dataflow(din0, dout, height, width, cycles);
}
