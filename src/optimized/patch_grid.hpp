#pragma once

// The image is cut into PH x PW patches, raster order (loop tiling: every
// process of a kernel walks the same grid, one patch after another). The last
// row and column of patches may stick out of the image: read_rows fills them
// by repeating the edge, store_scores drops what is outside.
//
// The blocks' LOOP_TRIPCOUNTs only feed the latency estimates in the csynth
// report; they assume an image of up to 19 x 35 patches of 32 x 32.
template<int PH>
int patch_rows(int height) { return (height + PH - 1) / PH; }

template<int PW>
int patch_cols(int width) { return (width + PW - 1) / PW; }

template<int PH, int PW>
int patch_count(int height, int width) { return patch_rows<PH>(height) * patch_cols<PW>(width); }
