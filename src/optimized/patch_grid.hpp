#pragma once

#include <stdint.h>
#include "fixed_point.hpp"

// Every process walks the image as the same grid of patches, in raster order.
// The last row and column of patches may stick out of the image: read_rows
// repeats the edge, store_scores drops what is outside. The index arithmetic
// is 16-bit: at most 65535 pixels a side.
template<int PATCH_HEIGHT>
int patch_rows(int height) { return (height + PATCH_HEIGHT - 1) / PATCH_HEIGHT; }

template<int PATCH_WIDTH>
int patch_cols(int width) { return (width + PATCH_WIDTH - 1) / PATCH_WIDTH; }

template<int PATCH_HEIGHT, int PATCH_WIDTH>
int patch_count(int height, int width) {
    return mul<false, int>((uint16_t)patch_rows<PATCH_HEIGHT>(height), (uint16_t)patch_cols<PATCH_WIDTH>(width));
}

// The pixels a per-pixel process walks: PIXELS of every patch (its map). MIN and
// MAX, a patch and the largest capture's 665, are for HLS's reports only.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int PIXELS>
struct patch_pixels {
    static constexpr int MIN = PIXELS, MAX = 665 * PIXELS;
    static uint32_t count(int height, int width) {
        #pragma HLS INLINE
        return patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * PIXELS;
    }
};
