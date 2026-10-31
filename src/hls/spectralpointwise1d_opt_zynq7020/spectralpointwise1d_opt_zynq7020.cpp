#include "spectralpointwise1d_opt_zynq7020.hpp"
#include "../spectralpointwise1d_opt/spectralpointwise1d_opt_lanes.hpp"

void spectralpointwise1d_opt_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                                      hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles)
{
    // Two readers on HP0-1, the writer on HP2. depth is for co-simulation only.
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
    #pragma HLS DATAFLOW
    hls::stream<cube_word_t> cube_words[LANES];
    hls::stream<vec_t<uact_t, 24>> convsp1_out[LANES];
    hls::stream<vec_t<uact_t, 12>> convsp2_out[LANES];
    hls::stream<vec_t<acc_t, CLASSES>> scores[LANES];
    hls::stream<bool> all_written("all_written");
    // The store takes the lanes in turn, a block each: each lane's scores need
    // room for the blocks it finishes meanwhile.
    #pragma HLS STREAM variable=cube_words depth=64
    #pragma HLS STREAM variable=scores     depth=32

    read_pixels_ahead<WORDS_PER_PIXEL, BLOCK_PIXELS, LANES, 0>(din0, cube_words[0], height, width);
    lane_convsp1<0>(cube_words[0], convsp1_out[0], height, width);
    lane_convsp2<0>(convsp1_out[0], convsp2_out[0], height, width);
    lane_convsp3<0>(convsp2_out[0], scores[0], height, width);
    read_pixels_ahead<WORDS_PER_PIXEL, BLOCK_PIXELS, LANES, 1>(din1, cube_words[1], height, width);
    lane_convsp1<1>(cube_words[1], convsp1_out[1], height, width);
    lane_convsp2<1>(convsp1_out[1], convsp2_out[1], height, width);
    lane_convsp3<1>(convsp2_out[1], scores[1], height, width);
    store_pixels_lanes<CLASSES, LANES, BLOCK_PIXELS, 64, 8>(scores, dout, height, width, all_written);

    count_cycles(all_written, cycles);   // last: C simulation runs the processes in order
}
