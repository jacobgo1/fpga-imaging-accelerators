#pragma once

// patch_buffer for the Zynq-7030: the 7020's (z-score, ping-pong buffer in block
// RAM, several words per chunk), since neither chip has UltraRAM. The 7030's
// kernel uses it with more ports and larger chunks: 4 ports x 3 words a chunk
// is 12 banks of 10 chunks x 8 rows x 32 pixels x 2 buffers x 32 bits, about
// 60 of its 265 BRAM36.
#include "../zynq7020/patch_buffer.hpp"
