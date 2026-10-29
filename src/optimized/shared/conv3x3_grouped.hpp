#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// A 3x3 conv sized for any time a pixel: IN_PER_CYCLE input channels times
// OUT_PER_CYCLE output channels times ROWS_PER_STEP rows of the window a cycle
// (1 for a chip with few DSPs, 3 for the whole window). The same streams as
// conv3x3_narrow; out the sums, or with RELU_SHIFT > 0 ReLU'd and rounded to
// activations, so no ReLU module is needed after it.
//
// A pixel's steps: output group outer, then input group, then window rows.
// The weights are one wide memory word a step: one memory a layer, since a
// thousand small ones made sp_unet_large_opt_direct too congested to route.
template<int IN_PER_CYCLE, int OUT_PER_CYCLE, int ROWS_PER_STEP = 3>
struct grouped_word_t {
    wgt_t v[OUT_PER_CYCLE][IN_PER_CYCLE][ROWS_PER_STEP][3];
};

template<int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE,
         int ROWS_PER_STEP, int RELU_SHIFT, bool DSP>
void conv3x3_grouped_patch(
    hls::stream<vec_t<act_t, IN_CHANNELS> >& in,
    const grouped_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, ROWS_PER_STEP>
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE) * (3 / ROWS_PER_STEP)],
    const acc_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE],
    hls::stream<vec_t<typename sums_out<RELU_SHIFT>::type, OUT_CHANNELS> >& out) {
    constexpr int IN_GROUPS = IN_CHANNELS / IN_PER_CYCLE, OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    constexpr int ROW_STEPS = 3 / ROWS_PER_STEP, STEPS = OUT_GROUPS * IN_GROUPS * ROW_STEPS;
    constexpr int PIXELS = MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 1;
    typedef typename sums_out<RELU_SHIFT>::type out_t;

    act_t lines[2][MAP_WIDTH][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=3
    act_t window[3][3][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=window type=complete dim=0
    out_t outputs[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=outputs type=complete
    acc_t running[OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=running type=complete

    int in_col = 0;
    int y = 0, x = 0;                                   // the window's centre
    int out_group = 0, in_group = 0, row_step = 0;
    int inputs = 0;                                     // input steps so far, the last W + 1 without input

    conv: for (int t = 0; t < (PIXELS + LAG) * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=lines type=inter false
        // Spelled out so that with one row step they are constants.
        const bool first_row = ROW_STEPS == 1 || row_step == 0;
        const bool last_row = ROW_STEPS == 1 || row_step == ROW_STEPS - 1;
        const bool first_step = out_group == 0 && in_group == 0 && first_row;
        const bool last_input = in_group == IN_GROUPS - 1 && last_row;
        const bool last_step = out_group == OUT_GROUPS - 1 && last_input;

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
            const int first_window_row = ROW_STEPS == 1 ? 0 : row_step;
            const grouped_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, ROWS_PER_STEP> word =
                weight[(out_group * IN_GROUPS + in_group) * ROW_STEPS + (ROW_STEPS == 1 ? 0 : row_step)];
            for (int j = 0; j < OUT_PER_CYCLE; j++) {
                #pragma HLS UNROLL
                acc_t sum = 0;
                for (int i = 0; i < IN_PER_CYCLE; i++) {
                    #pragma HLS UNROLL
                    for (int r = 0; r < ROWS_PER_STEP; r++) {
                        #pragma HLS UNROLL
                        const int ky = first_window_row + r;
                        for (int kx = 0; kx < 3; kx++) {
                            #pragma HLS UNROLL
                            const act_t a = (row_inside[ky] && col_inside[kx])
                                          ? window[ky][kx][in_group * IN_PER_CYCLE + i] : (act_t)0;
                            sum += mul<DSP, acc_t>(a, word.v[j][i][r][kx]);
                        }
                    }
                }
                const acc_t total = (in_group == 0 && first_row ? bias[out_group][j] : running[j]) + sum;
                running[j] = total;
                // Rounded as its output group finishes: OUT_PER_CYCLE rounding units, not OUT_CHANNELS.
                const out_t done = sums_out<RELU_SHIFT>::of(total);
                for (int og = 0; og < OUT_GROUPS; og++) {
                    #pragma HLS UNROLL
                    if (last_input && out_group == og) outputs[og * OUT_PER_CYCLE + j] = done;
                }
            }
            if (last_step) {
                vec_t<out_t, OUT_CHANNELS> pixel_out;
                for (int c = 0; c < OUT_CHANNELS; c++) {
                    #pragma HLS UNROLL
                    pixel_out.v[c] = outputs[c];
                }
                out.write(pixel_out);
            }
        }

        if (!last_row) {
            row_step++;
        } else {
            row_step = 0;
            if (in_group < IN_GROUPS - 1) {
                in_group++;
            } else {
                in_group = 0;
                if (!last_step) {
                    out_group++;
                } else {
                    out_group = 0;
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
}

// The weights a word a step, read from the ROM once, when the layer starts.
template<int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE, int ROWS_PER_STEP, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS]>
void load_grouped_weights(
    grouped_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, ROWS_PER_STEP>
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE) * (3 / ROWS_PER_STEP)],
    acc_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE]) {
    constexpr int IN_GROUPS = IN_CHANNELS / IN_PER_CYCLE, ROW_STEPS = 3 / ROWS_PER_STEP;
    constexpr int STEPS = (OUT_CHANNELS / OUT_PER_CYCLE) * IN_GROUPS * ROW_STEPS;
    load_w: for (int step = 0; step < STEPS; step++) {
        #pragma HLS PIPELINE
        const int out_group = step / (IN_GROUPS * ROW_STEPS), in_group = (step / ROW_STEPS) % IN_GROUPS;
        const int row_step = step % ROW_STEPS;
        grouped_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, ROWS_PER_STEP> word;
        for (int j = 0; j < OUT_PER_CYCLE; j++) {
            #pragma HLS UNROLL
            for (int i = 0; i < IN_PER_CYCLE; i++) {
                #pragma HLS UNROLL
                for (int r = 0; r < ROWS_PER_STEP; r++) {
                    #pragma HLS UNROLL
                    for (int kx = 0; kx < 3; kx++) {
                        #pragma HLS UNROLL
                        word.v[j][i][r][kx] = WEIGHT[out_group * OUT_PER_CYCLE + j][in_group * IN_PER_CYCLE + i]
                                                    [row_step * ROWS_PER_STEP + r][kx];
                    }
                }
            }
        }
        weight[step] = word;
    }
    load_b: for (int o = 0; o < OUT_CHANNELS; o++) {
        #pragma HLS PIPELINE II=1
        bias[o / OUT_PER_CYCLE][o % OUT_PER_CYCLE] = (acc_t)BIAS[o] * ((acc_t)1 << BIAS_SHIFT);
    }
}

// The weight memory is block RAM when deep, LUTs when shallow (block RAM would
// be mostly empty). impl takes only a literal word: one branch each.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int OUT_PER_CYCLE, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS], bool DSP,
         int ROWS_PER_STEP = 3, int RELU_SHIFT = 0>
void conv3x3_grouped(hls::stream<vec_t<act_t, IN_CHANNELS> >& in,
                     hls::stream<vec_t<typename sums_out<RELU_SHIFT>::type, OUT_CHANNELS> >& out,
                     int height, int width) {
    constexpr int STEPS = (OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE) * (3 / ROWS_PER_STEP);
    acc_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    if (STEPS > 64) {
        grouped_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, ROWS_PER_STEP> weight[STEPS];
        #pragma HLS AGGREGATE variable=weight compact=bit
        #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=bram
        load_grouped_weights<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, ROWS_PER_STEP, BIAS_SHIFT,
                             WEIGHT, BIAS>(weight, bias);
        patches_bram: for (int k = 0; k < patches; k++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=665
            conv3x3_grouped_patch<MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE,
                                  ROWS_PER_STEP, RELU_SHIFT, DSP>(in, weight, bias, out);
        }
    } else {
        grouped_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, ROWS_PER_STEP> weight[STEPS];
        #pragma HLS AGGREGATE variable=weight compact=bit
        #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=lutram
        load_grouped_weights<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, ROWS_PER_STEP, BIAS_SHIFT,
                             WEIGHT, BIAS>(weight, bias);
        patches: for (int k = 0; k < patches; k++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=665
            conv3x3_grouped_patch<MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE,
                                  ROWS_PER_STEP, RELU_SHIFT, DSP>(in, weight, bias, out);
        }
    }
}
