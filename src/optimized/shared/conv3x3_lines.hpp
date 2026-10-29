#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// A 3x3 conv for a chip short of flip-flops: the window lives in memory, not
// registers. The same streams and answers as conv3x3_grouped. The last three
// input rows are three memories (row r in memory r mod 3); a step reads one
// window column for IN_PER_CYCLE channels and uses TAPS_PER_STEP of its taps
// (3: the column, 1: one tap) times OUT_PER_CYCLE output channels.
//
// The input pixel is written to its row's memory a group a step. The window's
// centre trails the input by a row and two pixels, so a memory's write and
// read never meet. Taps outside the image are zeroed by the centre's position.
template<int IN_PER_CYCLE, int OUT_PER_CYCLE, int TAPS_PER_STEP>
struct lines_word_t {
    wgt_t v[OUT_PER_CYCLE][IN_PER_CYCLE][TAPS_PER_STEP];
};

template<int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE,
         int TAPS_PER_STEP, int RELU_SHIFT, bool DSP>
void conv3x3_lines_patch(
    hls::stream<vec_t<act_t, IN_CHANNELS> >& in,
    const lines_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, TAPS_PER_STEP>
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE) * (9 / TAPS_PER_STEP)],
    const acc_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE],
    hls::stream<vec_t<typename sums_out<RELU_SHIFT>::type, OUT_CHANNELS> >& out) {
    constexpr int IN_GROUPS = IN_CHANNELS / IN_PER_CYCLE, OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    constexpr int TAP_STEPS = 9 / TAPS_PER_STEP, STEPS = OUT_GROUPS * IN_GROUPS * TAP_STEPS;
    constexpr int PIXELS = MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 2;
    typedef typename sums_out<RELU_SHIFT>::type out_t;

    // [row memory][channel lane][column * IN_GROUPS + group]: plain arrays. A local
    // struct packed with AGGREGATE made Vitis HLS 2025.2 crash; reshaping the lanes
    // into one word made each lane's write its own cycle, 5-13x slower.
    act_t rows[3][IN_PER_CYCLE][MAP_WIDTH * IN_GROUPS];
    #pragma HLS ARRAY_PARTITION variable=rows type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=rows type=complete dim=2
    act_t pixel_in[IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=pixel_in type=complete
    out_t outputs[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=outputs type=complete
    acc_t running[OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=running type=complete

    int in_col = 0, in_memory = 0;   // the input's column and its row's memory
    int y = 0, x = 0;                // the window's centre
    int top_memory = 2;              // the memory of the row above the centre
    // Counters, not tap / 3 and tap % 3, which HLS built as a 36-cycle divider.
    int out_group = 0, in_group = 0, tap_col = 0, tap_row = 0, word_address = 0;
    int inputs = 0;                  // input pixels so far, the last W + 2 without input

    conv: for (int t = 0; t < (PIXELS + LAG) * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=rows type=inter false
        const bool first_tap = tap_col == 0 && (TAPS_PER_STEP == 3 || tap_row == 0);
        const bool last_tap = tap_col == 2 && (TAPS_PER_STEP == 3 || tap_row == 2);
        const bool first_step = out_group == 0 && in_group == 0 && first_tap;
        const bool last_input = in_group == IN_GROUPS - 1 && last_tap;
        const bool last_step = out_group == OUT_GROUPS - 1 && last_input;
        const bool have = inputs < PIXELS;

        if (first_step && have) {
            const vec_t<act_t, IN_CHANNELS> pixel = in.read();
            for (int ch = 0; ch < IN_CHANNELS; ch++) {
                #pragma HLS UNROLL
                pixel_in[ch] = pixel.v[ch];
            }
        }
        if (out_group == 0 && first_tap && have) {
            // Each memory has one write at a fixed place; the run-time choice is which runs.
            for (int i = 0; i < IN_PER_CYCLE; i++) {
                #pragma HLS UNROLL
                const act_t value = pixel_in[(IN_GROUPS == 1 ? 0 : in_group * IN_PER_CYCLE) + i];
                for (int m = 0; m < 3; m++) {
                    #pragma HLS UNROLL
                    if (in_memory == m) rows[m][i][in_col * IN_GROUPS + in_group] = value;
                }
            }
        }

        if (inputs >= LAG) {
            const int col = x + tap_col - 1;
            const int address = (col < 0 ? 0 : col > MAP_WIDTH - 1 ? MAP_WIDTH - 1 : col) * IN_GROUPS + in_group;
            act_t from_rows[3][IN_PER_CYCLE];
            #pragma HLS ARRAY_PARTITION variable=from_rows type=complete dim=0
            for (int m = 0; m < 3; m++) {
                #pragma HLS UNROLL
                for (int i = 0; i < IN_PER_CYCLE; i++) {
                    #pragma HLS UNROLL
                    from_rows[m][i] = rows[m][i][address];
                }
            }
            const bool row_inside[3] = {y > 0, true, y < MAP_HEIGHT - 1};
            const bool col_inside = (tap_col != 0 || x > 0) && (tap_col != 2 || x < MAP_WIDTH - 1);
            // Rows y - 1, y, y + 1 are in memories top_memory, + 1, + 2 (mod 3).
            act_t taps[TAPS_PER_STEP][IN_PER_CYCLE];
            #pragma HLS ARRAY_PARTITION variable=taps type=complete dim=0
            for (int r = 0; r < TAPS_PER_STEP; r++) {
                #pragma HLS UNROLL
                const int ky = TAPS_PER_STEP == 3 ? r : tap_row;
                const int m = top_memory + ky >= 3 ? top_memory + ky - 3 : top_memory + ky;
                for (int i = 0; i < IN_PER_CYCLE; i++) {
                    #pragma HLS UNROLL
                    const act_t value = m == 0 ? from_rows[0][i] : m == 1 ? from_rows[1][i] : from_rows[2][i];
                    taps[r][i] = (row_inside[ky] && col_inside) ? value : (act_t)0;
                }
            }
            const lines_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, TAPS_PER_STEP> word = weight[word_address];
            for (int j = 0; j < OUT_PER_CYCLE; j++) {
                #pragma HLS UNROLL
                acc_t sum = 0;
                for (int r = 0; r < TAPS_PER_STEP; r++) {
                    #pragma HLS UNROLL
                    for (int i = 0; i < IN_PER_CYCLE; i++) {
                        #pragma HLS UNROLL
                        sum += mul<DSP, acc_t>(taps[r][i], word.v[j][i][r]);
                    }
                }
                const acc_t total = (in_group == 0 && first_tap ? bias[out_group][j] : running[j]) + sum;
                running[j] = total;
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

        // Next: the window's taps, then the input group, the output group, the pixel.
        word_address = last_step ? 0 : word_address + 1;
        if (tap_col < 2) {
            tap_col++;
        } else if (TAPS_PER_STEP == 1 && tap_row < 2) {
            tap_col = 0;
            tap_row++;
        } else {
            tap_col = 0;
            tap_row = 0;
            if (in_group < IN_GROUPS - 1) {
                in_group++;
            } else {
                in_group = 0;
                if (!last_step) {
                    out_group++;
                } else {
                    out_group = 0;
                    if (in_col < MAP_WIDTH - 1) in_col++;
                    else { in_col = 0; in_memory = in_memory == 2 ? 0 : in_memory + 1; }
                    if (inputs >= LAG) {
                        if (x < MAP_WIDTH - 1) x++;
                        else { x = 0; top_memory = top_memory == 2 ? 0 : top_memory + 1; y = y < MAP_HEIGHT - 1 ? y + 1 : 0; }
                    }
                    inputs++;
                }
            }
        }
    }
}

// The weights a word a step, in the order the steps take them; nested loops, so
// no dividers even at load time.
template<int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE, int TAPS_PER_STEP, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS]>
void load_lines_weights(
    lines_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, TAPS_PER_STEP>
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE) * (9 / TAPS_PER_STEP)],
    acc_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE]) {
    constexpr int IN_GROUPS = IN_CHANNELS / IN_PER_CYCLE, TAP_STEPS = 9 / TAPS_PER_STEP;
    int step = 0;
    load_w: for (int out_group = 0; out_group < OUT_CHANNELS / OUT_PER_CYCLE; out_group++)
        for (int in_group = 0; in_group < IN_GROUPS; in_group++)
            for (int tap = 0; tap < TAP_STEPS; tap++) {
                #pragma HLS PIPELINE
                lines_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, TAPS_PER_STEP> word;
                for (int j = 0; j < OUT_PER_CYCLE; j++) {
                    #pragma HLS UNROLL
                    for (int i = 0; i < IN_PER_CYCLE; i++) {
                        #pragma HLS UNROLL
                        for (int r = 0; r < TAPS_PER_STEP; r++) {
                            #pragma HLS UNROLL
                            // A column a step: rows 0-2 of column tap. A tap a step: row tap / 3, column tap % 3.
                            const int ky = TAPS_PER_STEP == 3 ? r : tap >= 6 ? 2 : tap >= 3 ? 1 : 0;
                            const int kx = TAPS_PER_STEP == 3 ? tap : tap - 3 * ky;
                            word.v[j][i][r] = WEIGHT[out_group * OUT_PER_CYCLE + j][in_group * IN_PER_CYCLE + i][ky][kx];
                        }
                    }
                }
                weight[step] = word;
                step++;
            }
    load_b: for (int o = 0; o < OUT_CHANNELS; o++) {
        #pragma HLS PIPELINE II=1
        bias[o / OUT_PER_CYCLE][o % OUT_PER_CYCLE] = (acc_t)BIAS[o] * ((acc_t)1 << BIAS_SHIFT);
    }
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int OUT_PER_CYCLE, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS], bool DSP,
         int TAPS_PER_STEP, int RELU_SHIFT>
void conv3x3_lines(hls::stream<vec_t<act_t, IN_CHANNELS> >& in,
                   hls::stream<vec_t<typename sums_out<RELU_SHIFT>::type, OUT_CHANNELS> >& out,
                   int height, int width) {
    constexpr int STEPS = (OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE) * (9 / TAPS_PER_STEP);
    lines_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, TAPS_PER_STEP> weight[STEPS];
    #pragma HLS AGGREGATE variable=weight compact=bit
    #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=bram
    acc_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    load_lines_weights<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, TAPS_PER_STEP, BIAS_SHIFT,
                       WEIGHT, BIAS>(weight, bias);
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_lines_patch<MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE,
                            TAPS_PER_STEP, RELU_SHIFT, DSP>(in, weight, bias, out);
    }
}
