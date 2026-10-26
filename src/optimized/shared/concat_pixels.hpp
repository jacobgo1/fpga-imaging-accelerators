#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// concat_pixels: a U-Net's skip connection, torch.cat([a, b], dim=1) on two
// maps in raster order: each pixel's CA channels of a, then its CB of b. The
// order matters: the next conv's weights expect it (concat_golden.hpp). One
// pixel a cycle. The image is cut into PH x PW patches (patch_grid.hpp); PIX
// is the maps' pixels in a patch.
template<int PH, int PW, int PIX, int CA, int CB>
void concat_pixels(hls::stream<vec_t<act_t, CA> >& a, hls::stream<vec_t<act_t, CB> >& b,
                   hls::stream<vec_t<act_t, CA + CB> >& out, int height, int width) {
    constexpr int MAX_PIX = 665 * PIX;
    (void)MAX_PIX;   // only the pragma uses it
    const int n = patch_count<PH, PW>(height, width) * PIX;
    px: for (int k = 0; k < n; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=PIX max=MAX_PIX
        const vec_t<act_t, CA> va = a.read();
        const vec_t<act_t, CB> vb = b.read();
        vec_t<act_t, CA + CB> o;
        for (int c = 0; c < CA; c++) {
            #pragma HLS UNROLL
            o.v[c] = va.v[c];
        }
        for (int c = 0; c < CB; c++) {
            #pragma HLS UNROLL
            o.v[CA + c] = vb.v[c];
        }
        out.write(o);
    }
}
