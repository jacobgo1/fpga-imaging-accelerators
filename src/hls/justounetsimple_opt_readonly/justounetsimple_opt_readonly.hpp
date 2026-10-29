#ifndef JUSTOUNETSIMPLE_OPT_READONLY_HLS_HPP
#define JUSTOUNETSIMPLE_OPT_READONLY_HLS_HPP

// justounetsimple_opt's four readers with the compute replaced by a checksum:
// it times the DDR reads alone. dout gets one word, each band's sum mod 2^16,
// band b in the low (b even) or high half of int32 b / 2.
#include "../justounetsimple_opt/justounetsimple_opt.hpp"

void justounetsimple_opt_readonly(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                                  hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                                  hls::burst_maxi<score_word_t> dout, int height, int width,
                                  uint32_t& cycles);

#endif
