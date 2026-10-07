#pragma once

#include "conv3x3_stream.hpp"

// What happens between two convolutions. Each reads a conv3x3_stream's output
// (all channels of a pixel, raster order) and writes activations in the same
// form, the order the next conv3x3_stream with P = all its channels reads.
//
// Sums come in with S fraction bits and leave as activations with S - SHIFT:
// rounded to nearest, ReLU'd, saturated at the int16 maximum. Rounding is
// monotonic, so max-pooling before or after it gives the same result.

// One sum -> one activation: ReLU, round off SHIFT bits, saturate.
template<int SHIFT>
act_t requant_relu(acc_t v) {
    #pragma HLS INLINE
    static_assert(SHIFT >= 1, "requant_relu drops at least one bit");
    if (v <= 0) return 0;
    const acc_t r = (v + ((acc_t)1 << (SHIFT - 1))) >> SHIFT;
    return r > 32767 ? (act_t)32767 : (act_t)r;
}

// Round off SHIFT bits (no ReLU, no saturation: the result only gets smaller).
template<int SHIFT>
acc_t round_shift(acc_t v) {
    #pragma HLS INLINE
    static_assert(SHIFT >= 0, "round_shift only drops bits");
    return SHIFT == 0 ? v : (v + ((acc_t)1 << (SHIFT > 0 ? SHIFT - 1 : 0))) >> SHIFT;
}

static inline act_t max2(act_t a, act_t b) {
    #pragma HLS INLINE
    return a > b ? a : b;
}

// ReLU, then 2x2 max-pool: H x W -> H/2 x W/2. A pooled pixel leaves as soon
// as its block's last value (odd row, odd column) arrives.
template<int H, int W, int CH, int SHIFT>
void relu_pool(hls::stream<vec_t<acc_t, CH> >& in, hls::stream<vec_t<act_t, CH> >& out) {
    act_t row[W / 2][CH];   // the even row's pairs, pooled
    #pragma HLS ARRAY_PARTITION variable=row type=complete dim=2
    act_t left[CH];         // the even column's value
    #pragma HLS ARRAY_PARTITION variable=left type=complete
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=row type=inter false
            const vec_t<acc_t, CH> v = in.read();
            vec_t<act_t, CH> o;
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                const act_t a = requant_relu<SHIFT>(v.v[ch]);
                if (x % 2 == 0) {
                    left[ch] = a;
                } else {
                    const act_t m = max2(left[ch], a);
                    if (y % 2 == 0) row[x / 2][ch] = m;
                    else o.v[ch] = max2(row[x / 2][ch], m);
                }
            }
            if (y % 2 == 1 && x % 2 == 1) out.write(o);
        }
}

// ReLU, 2x2 max-pool, then 2x nearest upsample: H x W -> H x W, every 2x2
// block replaced by its maximum. Each pair of rows is taken in, then given out.
template<int H, int W, int CH, int SHIFT>
void relu_pool_up(hls::stream<vec_t<acc_t, CH> >& in, hls::stream<vec_t<act_t, CH> >& out) {
    act_t blk[W / 2][CH];   // the maxima of this row pair's 2x2 blocks
    #pragma HLS ARRAY_PARTITION variable=blk type=complete dim=2
    act_t left[CH];
    #pragma HLS ARRAY_PARTITION variable=left type=complete
    pairs: for (int j = 0; j < H / 2; j++) {
        take: for (int i = 0; i < 2 * W; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=blk type=inter false
            const int x = i % W;
            const vec_t<acc_t, CH> v = in.read();
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                const act_t a = requant_relu<SHIFT>(v.v[ch]);
                if (x % 2 == 0) {
                    left[ch] = a;
                } else {
                    const act_t m = max2(left[ch], a);
                    blk[x / 2][ch] = (i < W) ? m : max2(blk[x / 2][ch], m);
                }
            }
        }
        give: for (int i = 0; i < 2 * W; i++) {
            #pragma HLS PIPELINE II=1
            vec_t<act_t, CH> o;
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                o.v[ch] = blk[(i % W) / 2][ch];
            }
            out.write(o);
        }
    }
}

// ReLU, then 2x nearest upsample: H x W -> 2H x 2W. Each row is taken in, then
// given out twice, every pixel twice.
template<int H, int W, int CH, int SHIFT>
void relu_up(hls::stream<vec_t<acc_t, CH> >& in, hls::stream<vec_t<act_t, CH> >& out) {
    act_t line[W][CH];
    #pragma HLS ARRAY_PARTITION variable=line type=complete dim=2
    rows: for (int y = 0; y < H; y++) {
        take: for (int x = 0; x < W; x++) {
            #pragma HLS PIPELINE II=1
            const vec_t<acc_t, CH> v = in.read();
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                line[x][ch] = requant_relu<SHIFT>(v.v[ch]);
            }
        }
        give: for (int i = 0; i < 4 * W; i++) {
            #pragma HLS PIPELINE II=1
            vec_t<act_t, CH> o;
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                o.v[ch] = line[(i % (2 * W)) / 2][ch];
            }
            out.write(o);
        }
    }
}
