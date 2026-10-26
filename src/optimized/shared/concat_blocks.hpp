#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// concat_blocks: a U-Net's skip connection, torch.cat([a, b], dim=1) on two
// maps in 2x2-block order (wino_conv3x3.hpp): each pixel's CA channels of a,
// then its CB of b. The order matters: the next conv's weights expect it
// (concat_golden.hpp). One block a cycle.
//
// The image is cut into PH x PW patches (patch_grid.hpp); H x W is the maps'
// size in a patch (pixels).
template<int PH, int PW, int H, int W, int CA, int CB>
void concat_blocks(hls::stream<vec_t<act_t, 4 * CA> >& a, hls::stream<vec_t<act_t, 4 * CB> >& b,
                   hls::stream<vec_t<act_t, 4 * (CA + CB)> >& out, int height, int width) {
    constexpr int BLOCKS = (H / 2) * (W / 2), MAX_BLOCKS = 665 * BLOCKS;
    (void)BLOCKS; (void)MAX_BLOCKS;   // only the pragma uses them
    const int n = patch_count<PH, PW>(height, width) * BLOCKS;
    blocks: for (int k = 0; k < n; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=BLOCKS max=MAX_BLOCKS
        const vec_t<act_t, 4 * CA> va = a.read();
        const vec_t<act_t, 4 * CB> vb = b.read();
        vec_t<act_t, 4 * (CA + CB)> o;
        for (int q = 0; q < 4; q++) {
            #pragma HLS UNROLL
            for (int c = 0; c < CA; c++) {
                #pragma HLS UNROLL
                o.v[q * (CA + CB) + c] = va.v[q * CA + c];
            }
            for (int c = 0; c < CB; c++) {
                #pragma HLS UNROLL
                o.v[q * (CA + CB) + CA + c] = vb.v[q * CB + c];
            }
        }
        out.write(o);
    }
}
