#ifndef JUSTOUNETSIMPLE_OPT_READONLY_HLS_HPP
#define JUSTOUNETSIMPLE_OPT_READONLY_HLS_HPP

// justounetsimple_opt with the compute taken out, to measure the DDR reads
// alone (ZCU104). The four readers are justounetsimple_opt's, unchanged: the
// same ports, the same bursts, the same rows in the same order. In place of
// patch_buffer and the convolutions, one process (drain) takes a word from
// every reader each cycle, as patch_buffer does, and never waits for anything
// else:
//
//   read_rows_ahead<0> ─┐
//   read_rows_ahead<1> ─┤
//   read_rows_ahead<2> ─┼─► drain ─► dout: one word, the checksum
//   read_rows_ahead<3> ─┘
//
// So its time is how fast the readers get the cube out of DDR. If it is about
// justounetsimple_opt's time on the same image, the reads are what limit it.
// Ideal: each port reads 8 rows x 32 pixels x 15 words of a patch, 3 840
// cycles at a word per cycle; justounetsimple_opt's conv1 needs 4 129.
//
// drain adds every word up, so that HLS keeps every read and the notebook can
// check that each word arrived: each of a word's 8 bands in its own uint16
// lane, mod 2^16, the edge patches' padding included.
//
// Ports and registers as justounetsimple_opt's: din0 .. din3 the same raw cube,
// [height][width][120] uint16. dout: one 128-bit word, the 8 lanes, lane 2i in
// the low half of int32 i. cycles: as justounetsimple_opt's, the kernel's time
// in clock cycles, until the checksum is in memory (0 in C simulation).
#include "../justounetsimple_opt/justounetsimple_opt.hpp"

void justounetsimple_opt_readonly(hls::burst_maxi<jopt_raw_t> din0, hls::burst_maxi<jopt_raw_t> din1,
                                  hls::burst_maxi<jopt_raw_t> din2, hls::burst_maxi<jopt_raw_t> din3,
                                  hls::burst_maxi<jopt_out_t> dout, int height, int width,
                                  uint32_t& cycles);

#endif // JUSTOUNETSIMPLE_OPT_READONLY_HLS_HPP
