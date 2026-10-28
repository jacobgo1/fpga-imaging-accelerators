#pragma once

// The optimized 1D-SpectralPointwise's DATAFLOW chain in four lanes, shared by
// spectralpointwise1d_opt (ZCU104) and spectralpointwise1d_opt_zynq7030 (the same
// lanes on 64-bit ports: 4 bands a word, 30 words a pixel). Before including it,
// a kernel's .cpp includes its own header. spectralpointwise1d_opt.hpp has the
// design, spectralpointwise1d_opt_lanes.hpp the layers of a lane.

#include "spectralpointwise1d_opt_lanes.hpp"

static_assert(LANES == 4 && BLOCK <= 16, "this chain has four lanes; s_scores holds two blocks");

static void spectralpointwise1d_opt_dataflow(hls::burst_maxi<spw_raw_t> din0, hls::burst_maxi<spw_raw_t> din1,
                                             hls::burst_maxi<spw_raw_t> din2, hls::burst_maxi<spw_raw_t> din3,
                                             hls::burst_maxi<spw_out_t> dout, int height, int width,
                                             uint32_t& cycles)
{
#pragma HLS DATAFLOW
    hls::stream<spw_raw_t> s_raw[LANES];
    hls::stream<vec_t<uact_t, 24>> s_h1[LANES];
    hls::stream<vec_t<uact_t, 12>> s_h2[LANES];
    hls::stream<vec_t<acc_t, CLASSES>> s_scores[LANES];
    hls::stream<bool> s_finished("s_finished");
// A reader's bursts arrive in blocks; let it run ahead. The store takes the
// lanes in turn, a block each: while it waits for one lane's block the others
// finish theirs, so each lane's scores need room for a block and then some:
// 32 is two of the ZCU104's blocks, four of the 7030's.
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
    read_pixels_ahead<WORDS, BLOCK, LANES, 2>(din2, s_raw[2], height, width);
    spw_convsp1<2>(s_raw[2], s_h1[2], height, width);
    spw_convsp2<2>(s_h1[2], s_h2[2], height, width);
    spw_convsp3<2>(s_h2[2], s_scores[2], height, width);
    read_pixels_ahead<WORDS, BLOCK, LANES, 3>(din3, s_raw[3], height, width);
    spw_convsp1<3>(s_raw[3], s_h1[3], height, width);
    spw_convsp2<3>(s_h1[3], s_h2[3], height, width);
    spw_convsp3<3>(s_h2[3], s_scores[3], height, width);
    store_pixels_lanes<CLASSES, LANES, BLOCK, 64, 8>(s_scores, dout, height, width, s_finished);
    // Last: C simulation runs the processes in this order (count_cycles.hpp).
    count_cycles(s_finished, cycles);
}
