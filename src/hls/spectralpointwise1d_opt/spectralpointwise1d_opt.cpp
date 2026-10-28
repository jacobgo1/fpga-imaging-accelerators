// spectralpointwise1d_opt: the optimized 1D-SpectralPointwise on a whole raw capture.
// The design is in spectralpointwise1d_opt.hpp, the processes in spectralpointwise1d_opt_dataflow.hpp.
#include "spectralpointwise1d_opt.hpp"
#include "spectralpointwise1d_opt_dataflow.hpp"

void spectralpointwise1d_opt(hls::burst_maxi<spw_raw_t> din0, hls::burst_maxi<spw_raw_t> din1,
                             hls::burst_maxi<spw_raw_t> din2, hls::burst_maxi<spw_raw_t> din3,
                             hls::burst_maxi<spw_out_t> dout, int height, int width, uint32_t& cycles)
{
// Five AXI masters, each on its own port to DDR (boards/zcu104/system.tcl): four
// read the cube, a lane each, blocks of 16 pixels in turn, through HP0-3; one
// writes the scores through HPC0. depth is only for co-simulation: the
// testbench's 9 x 7 image.
#pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=945 max_read_burst_length=256 num_read_outstanding=4
#pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=945 max_read_burst_length=256 num_read_outstanding=4
#pragma HLS INTERFACE mode=m_axi port=din2 bundle=gmem2 offset=slave depth=945 max_read_burst_length=256 num_read_outstanding=4
#pragma HLS INTERFACE mode=m_axi port=din3 bundle=gmem3 offset=slave depth=945 max_read_burst_length=256 num_read_outstanding=4
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
