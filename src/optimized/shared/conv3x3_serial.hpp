#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// A 3x3 conv for a chip with few DSPs: IN_PER_CYCLE input channels times one output
// channel a cycle, IN_PER_CYCLE x 9 multipliers whatever the layer's size. The same
// streams as conv3x3_narrow, IN_CHANNELS / IN_PER_CYCLE times slower and smaller.
template<int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, bool DSP>
void conv3x3_serial_patch(hls::stream<vec_t<act_t, IN_CHANNELS> >& in,
                          const wgt_t weight[OUT_CHANNELS][IN_CHANNELS / IN_PER_CYCLE][IN_PER_CYCLE][3][3],
                          const acc_t bias[OUT_CHANNELS],
                          hls::stream<vec_t<acc_t, OUT_CHANNELS> >& out) {
    constexpr int GROUPS = IN_CHANNELS / IN_PER_CYCLE;
    constexpr int STEPS = OUT_CHANNELS * GROUPS;
    constexpr int PIXELS = MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 1;

    act_t lines[2][MAP_WIDTH][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=3
    act_t window[3][3][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=window type=complete dim=0
    acc_t sums[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=sums type=complete

    int in_col = 0;
    int y = 0, x = 0;               // the window's centre
    int o = 0, group = 0;           // this step's output channel and input group
    int inputs = 0;                 // input steps so far, the last W + 1 without input
    acc_t running = 0;

    conv: for (int t = 0; t < (PIXELS + LAG) * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=lines type=inter false
        const bool first_step = o == 0 && group == 0;
        const bool last_group = group == GROUPS - 1;
        const bool last_step = o == OUT_CHANNELS - 1 && last_group;

        if (first_step) {
            const bool have = inputs < PIXELS;
            vec_t<act_t, IN_CHANNELS> pixel;
            if (have) pixel = in.read();
            for (int ch = 0; ch < IN_CHANNELS; ch++) {
                #pragma HLS UNROLL
                const act_t two_up = lines[0][in_col][ch], one_up = lines[1][in_col][ch];
                const act_t now = have ? pixel.v[ch] : (act_t)0;
                lines[0][in_col][ch] = one_up;
                lines[1][in_col][ch] = now;
                for (int ky = 0; ky < 3; ky++) {
                    #pragma HLS UNROLL
                    window[ky][0][ch] = window[ky][1][ch];
                    window[ky][1][ch] = window[ky][2][ch];
                }
                window[0][2][ch] = two_up;
                window[1][2][ch] = one_up;
                window[2][2][ch] = now;
            }
            in_col = (in_col == MAP_WIDTH - 1) ? 0 : in_col + 1;
        }

        if (inputs >= LAG) {
            const bool row_inside[3] = {y > 0, true, y < MAP_HEIGHT - 1};
            const bool col_inside[3] = {x > 0, true, x < MAP_WIDTH - 1};
            acc_t sum = 0;
            for (int i = 0; i < IN_PER_CYCLE; i++) {
                #pragma HLS UNROLL
                for (int ky = 0; ky < 3; ky++) {
                    #pragma HLS UNROLL
                    for (int kx = 0; kx < 3; kx++) {
                        #pragma HLS UNROLL
                        const act_t a = (row_inside[ky] && col_inside[kx])
                                      ? window[ky][kx][group * IN_PER_CYCLE + i] : (act_t)0;
                        sum += mul<DSP, acc_t>(a, weight[o][group][i][ky][kx]);
                    }
                }
            }
            const acc_t total = (group == 0 ? bias[o] : running) + sum;
            running = total;
            if (last_group) sums[o] = total;
            if (last_step) {
                vec_t<acc_t, OUT_CHANNELS> pixel_sums;
                for (int c = 0; c < OUT_CHANNELS; c++) {
                    #pragma HLS UNROLL
                    pixel_sums.v[c] = sums[c];
                }
                out.write(pixel_sums);
            }
        }

        if (!last_group) {
            group++;
        } else {
            group = 0;
            if (!last_step) {
                o++;
            } else {
                o = 0;
                if (inputs >= LAG) {
                    if (x < MAP_WIDTH - 1) x++;
                    else if (y < MAP_HEIGHT - 1) { x = 0; y++; }
                    else { x = 0; y = 0; }
                }
                inputs++;
            }
        }
    }
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS], bool DSP>
void conv3x3_serial(hls::stream<vec_t<act_t, IN_CHANNELS> >& in, hls::stream<vec_t<acc_t, OUT_CHANNELS> >& out,
                    int height, int width) {
    wgt_t weight[OUT_CHANNELS][IN_CHANNELS / IN_PER_CYCLE][IN_PER_CYCLE][3][3];
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=4
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=5
    acc_t bias[OUT_CHANNELS];

    load_w: for (int o = 0; o < OUT_CHANNELS; o++)
        for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    #pragma HLS PIPELINE II=1
                    weight[o][in_ch / IN_PER_CYCLE][in_ch % IN_PER_CYCLE][ky][kx] = WEIGHT[o][in_ch][ky][kx];
                }
    load_b: for (int o = 0; o < OUT_CHANNELS; o++) {
        #pragma HLS PIPELINE II=1
        bias[o] = (acc_t)BIAS[o] * ((acc_t)1 << BIAS_SHIFT);
    }
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_serial_patch<MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, DSP>(in, weight, bias, out);
    }
}
