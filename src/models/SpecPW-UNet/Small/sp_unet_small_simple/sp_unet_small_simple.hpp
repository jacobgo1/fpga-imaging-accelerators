#ifndef SP_UNET_SMALL_SIMPLE_HLS_HPP
#define SP_UNET_SMALL_SIMPLE_HLS_HPP

#include "sp_unet_simple.hpp"

// The argument names become the kernel's register names: keep them.
void sp_unet_small_simple(const cube_word_t* din0, score_word_t* dout, int height, int width);

#endif
