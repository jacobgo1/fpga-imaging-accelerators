#include "spectralpointwise1d_opt_zynq7030.hpp"
#include "../spectralpointwise1d_opt/spectralpointwise1d_opt_dataflow.hpp"

void spectralpointwise1d_opt_zynq7030(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                                      hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                                      hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles)
{
    // Four readers on HP0-3; the writer shares HP3. depth is for co-simulation only.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=1890 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=1890 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din2 bundle=gmem2 offset=slave depth=1890 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din3 bundle=gmem3 offset=slave depth=1890 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem4 offset=slave depth=63 max_write_burst_length=64 num_write_outstanding=16
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din2   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din3   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=cycles bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    spectralpointwise1d_opt_dataflow(din0, din1, din2, din3, dout, height, width, cycles);
}
