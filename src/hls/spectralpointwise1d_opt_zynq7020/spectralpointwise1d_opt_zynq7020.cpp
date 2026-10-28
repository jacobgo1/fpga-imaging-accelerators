// spectralpointwise1d_opt_zynq7020: the 1D-SpectralPointwise for the Zynq-7020 (ZedBoard) on a
// whole raw capture. The design is in spectralpointwise1d_opt_zynq7020.hpp, a lane's layers in
// ../spectralpointwise1d_opt/spectralpointwise1d_opt_lanes.hpp.
#include "spectralpointwise1d_opt_zynq7020.hpp"
#include "../spectralpointwise1d_opt/spectralpointwise1d_opt_lanes.hpp"

static_assert(LANES == 2 && BLOCK <= 16, "this chain has two lanes; s_scores holds two blocks");

static void spectralpointwise1d_opt_zynq7020_dataflow(hls::burst_maxi<spw_raw_t> din0, hls::burst_maxi<spw_raw_t> din1,
                                                      hls::burst_maxi<spw_out_t> dout, int height, int width,
                                                      uint32_t& cycles)
{
#pragma HLS DATAFLOW
    hls::stream<spw_raw_t> s_raw[LANES];
    hls::stream<vec_t<uact_t, 24>> s_h1[LANES];
    hls::stream<vec_t<uact_t, 12>> s_h2[LANES];
    hls::stream<vec_t<acc_t, CLASSES>> s_scores[LANES];
    hls::stream<bool> s_finished("s_finished");
// As in spectralpointwise1d_opt_dataflow.hpp: the readers run ahead, and each
// lane's scores have room for the blocks it finishes while the store takes the
// other lane's.
#pragma HLS STREAM variable = s_raw depth = 64
#pragma HLS STREAM variable = s_scores depth = 32

    read_pixels_ahead<WORDS, BLOCK, LANES, 0>(din0, s_raw[0], height, width);
    spw_convsp1<0>(s_raw[0], s_h1[0], height, width);
    spw_convsp2<0>(s_h1[0], s_h2[0], height, width);
    spw_convsp3<0>(s_h2[0], s_scores[0], height, width);
    read_pixels_ahead<WORDS, BLOCK, LANES, 1>(din1, s_raw[1], height, width);
    spw_convsp1<1>(s_raw[1], s_h1[1], height, width);
    spw_convsp2<1>(s_h1[1], s_h2[1], height, width);
    spw_convsp3<1>(s_h2[1], s_scores[1], height, width);
    store_pixels_lanes<CLASSES, LANES, BLOCK, 64, 8>(s_scores, dout, height, width, s_finished);
    // Last: C simulation runs the processes in this order (count_cycles.hpp).
    count_cycles(s_finished, cycles);
}

void spectralpointwise1d_opt_zynq7020(hls::burst_maxi<spw_raw_t> din0, hls::burst_maxi<spw_raw_t> din1,
                                      hls::burst_maxi<spw_out_t> dout, int height, int width, uint32_t& cycles)
{
// Three AXI masters, each on its own HP port (boards/zedboard/system.tcl): two read
// the cube, a lane each, blocks of 8 pixels in turn, through HP0 and HP1; one
// writes the scores through HP2. depth is only for co-simulation: the
// testbench's 9 x 7 image.
#pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=1890 max_read_burst_length=256 num_read_outstanding=4
#pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=1890 max_read_burst_length=256 num_read_outstanding=4
#pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem2 offset=slave depth=63 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=height bundle=control
#pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
#pragma HLS INTERFACE mode=s_axilite port=cycles bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    spectralpointwise1d_opt_zynq7020_dataflow(din0, din1, dout, height, width, cycles);
}
