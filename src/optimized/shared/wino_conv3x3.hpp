#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// A same-padded 3x3 conv and its ReLU as Winograd F(2x2, 3x3) (Lavin and Gray,
// 2016): a 2x2 block of outputs from the 4x4 tile around it, with 16 multiplies
// per input x output channel instead of 36:
//
//     4 Y = A^T [U * V] A,   V = B^T d B (the tile),   U = (2G) g (2G)^T (the weights)
//
// U is 4 times the textbook transform, so every coefficient is an integer and the
// result is exactly 4 x the direct conv's sum: bit for bit the same answer.
// The inputs are ReLU'd (0 .. 32767), so V fits 18 bits, U 12, their product one
// DSP48E2; sums are kept modulo 2^34, which leaves Y exact modulo 2^32.
//
// The maps are in 2x2-block order: a block is one vector of its 4 pixels,
// v[(2 dy + dx) * CHANNELS + c]. Then a maxpool window is a block, and so is
// Winograd's output.
//
// wino_up_conv3x3 is the same engine for an UpBlock (upsample by 2, then the conv).
// The upsampled tile's rows and columns are a, b, b, c of the low-res map, so V's
// row and column 2 are zero: 9 multiplies a block, from the low-res pixels.
//
// Each block is a period of IN_GROUPS x OUT_GROUPS steps, one a cycle. The window
// is four column slots; three are its left, centre and right columns, the fourth
// is filled for the next period from the two line buffers and the new input.
// The slots' roles rotate, and each memory has one write and one read at fixed
// places. Taps outside the image are zeroed by the centre's position.

#if defined(__VITIS_HLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_int<18> wino_input_t;     // a transformed input
typedef ap_int<12> wino_weight_t;    // a transformed weight (x 4)
typedef ap_int<30> wino_product_t;
typedef ap_int<34> wino_sum_t;       // modulo 2^34
#else
typedef int32_t wino_input_t;
typedef int16_t wino_weight_t;
typedef int32_t wino_product_t;
typedef int64_t wino_sum_t;          // exact; the low 32 bits of 4Y >> 2 are the same
#endif

template<int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE, bool UP>
struct wino_shape {
    static constexpr int IN_GROUPS = IN_CHANNELS / IN_PER_CYCLE, OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    static constexpr int STEPS = IN_GROUPS * OUT_GROUPS;
    static constexpr int PRODUCTS = UP ? 9 : 16;        // multiplies per input x output channel
    static constexpr int PIXELS_IN = UP ? 1 : 4;        // pixels per input vector
    static constexpr int COLUMN_VALUES = UP ? 3 : 8;    // a window column's values per channel
    static constexpr int ABOVE_VALUES = UP ? 1 : 2;     // line buffer: the pixel row above the block row
    static constexpr int ROW_VALUES = UP ? 1 : 4;       // line buffer: the block row itself
    static constexpr int WEIGHTS_PER_STEP = IN_PER_CYCLE * OUT_PER_CYCLE * PRODUCTS;
};

// The elements of V that are used: all 16, or (up) rows and columns 0, 1, 3.
constexpr int wino_row(bool up, int k) { return up ? (k / 3 == 2 ? 3 : k / 3) : k / 4; }
constexpr int wino_col(bool up, int k) { return up ? (k % 3 == 2 ? 3 : k % 3) : k % 4; }

// V = B^T d B, B^T = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]].
inline void wino_in_tile(const act_t tile[4][4], wino_input_t v[4][4]) {
    #pragma HLS INLINE
    int32_t t[4][4];
    for (int j = 0; j < 4; j++) {
        #pragma HLS UNROLL
        t[0][j] = tile[0][j] - tile[2][j];
        t[1][j] = tile[1][j] + tile[2][j];
        t[2][j] = tile[2][j] - tile[1][j];
        t[3][j] = tile[1][j] - tile[3][j];
    }
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        v[i][0] = t[i][0] - t[i][2];
        v[i][1] = t[i][1] + t[i][2];
        v[i][2] = t[i][2] - t[i][1];
        v[i][3] = t[i][1] - t[i][3];
    }
}

// The same for the upsampled tile of a low-res 3x3 neighbourhood: B^T on
// (a, b, b, c) is (a - b, 2b, 0, b - c).
inline void wino_in_up(const act_t neighbourhood[3][3], wino_input_t v[4][4]) {
    #pragma HLS INLINE
    int32_t t[4][3];
    for (int j = 0; j < 3; j++) {
        #pragma HLS UNROLL
        t[0][j] = neighbourhood[0][j] - neighbourhood[1][j];
        t[1][j] = 2 * neighbourhood[1][j];
        t[2][j] = 0;
        t[3][j] = neighbourhood[1][j] - neighbourhood[2][j];
    }
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        v[i][0] = t[i][0] - t[i][1];
        v[i][1] = 2 * t[i][1];
        v[i][2] = 0;
        v[i][3] = t[i][1] - t[i][2];
    }
}

// 4Y = A^T M A, A^T = [[1, 1, 1, 0], [0, 1, -1, -1]].
inline void wino_out(const wino_sum_t m[4][4], wino_sum_t y[2][2]) {
    #pragma HLS INLINE
    wino_sum_t t[2][4];
    for (int j = 0; j < 4; j++) {
        #pragma HLS UNROLL
        t[0][j] = m[0][j] + m[1][j] + m[2][j];
        t[1][j] = m[1][j] - m[2][j] - m[3][j];
    }
    for (int i = 0; i < 2; i++) {
        #pragma HLS UNROLL
        y[i][0] = t[i][0] + t[i][1] + t[i][2];
        y[i][1] = t[i][1] - t[i][2] - t[i][3];
    }
}

// U = (2G) g (2G)^T, 2G = [[2, 0, 0], [1, 1, 1], [1, -1, 1], [0, 0, 2]].
inline void wino_weight(const wgt_t g[3][3], int32_t u[4][4]) {
    #pragma HLS INLINE
    int32_t t[4][3];
    for (int col = 0; col < 3; col++) {
        #pragma HLS UNROLL
        t[0][col] = 2 * g[0][col];
        t[1][col] = g[0][col] + g[1][col] + g[2][col];
        t[2][col] = g[0][col] - g[1][col] + g[2][col];
        t[3][col] = 2 * g[2][col];
    }
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        u[i][0] = 2 * t[i][0];
        u[i][1] = t[i][0] + t[i][1] + t[i][2];
        u[i][2] = t[i][0] - t[i][1] + t[i][2];
        u[i][3] = 2 * t[i][2];
    }
}

// One step's transformed weights, [input][output][product], one wide memory word.
template<int WEIGHTS_PER_STEP>
struct wino_word_t {
    wino_weight_t v[WEIGHTS_PER_STEP];
};

template<int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE, bool UP, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS]>
void load_wino_weights(
    wino_word_t<wino_shape<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP>::WEIGHTS_PER_STEP>
        weight[wino_shape<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP>::STEPS],
    acc_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE]) {
    typedef wino_shape<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP> S;
    load_w: for (int step = 0; step < S::STEPS; step++) {
        #pragma HLS PIPELINE
        const int out_group = step / S::IN_GROUPS, in_group = step % S::IN_GROUPS;
        wino_word_t<S::WEIGHTS_PER_STEP> word;
        for (int i = 0; i < IN_PER_CYCLE; i++) {
            #pragma HLS UNROLL
            for (int j = 0; j < OUT_PER_CYCLE; j++) {
                #pragma HLS UNROLL
                int32_t u[4][4];
                wino_weight(WEIGHT[out_group * OUT_PER_CYCLE + j][in_group * IN_PER_CYCLE + i], u);
                for (int k = 0; k < S::PRODUCTS; k++) {
                    #pragma HLS UNROLL
                    word.v[(i * OUT_PER_CYCLE + j) * S::PRODUCTS + k] = u[wino_row(UP, k)][wino_col(UP, k)];
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

// The engine on all patches' GRID_HEIGHT x GRID_WIDTH centres (blocks, or low-res
// pixels). Out: a block of 4 pixels, ReLU'd and rounded off SHIFT bits.
template<int GRID_HEIGHT, int GRID_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE,
         bool UP, int SHIFT, bool DSP>
void wino_run(
    hls::stream<vec_t<act_t, wino_shape<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP>::PIXELS_IN
                             * IN_CHANNELS> >& in,
    const wino_word_t<wino_shape<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP>::WEIGHTS_PER_STEP>
        weight[wino_shape<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP>::STEPS],
    const acc_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE],
    hls::stream<vec_t<act_t, 4 * OUT_CHANNELS> >& out, int patches) {
    typedef wino_shape<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP> S;
    constexpr int IN_GROUPS = S::IN_GROUPS, OUT_GROUPS = S::OUT_GROUPS, PRODUCTS = S::PRODUCTS;
    constexpr int PIXELS_IN = S::PIXELS_IN;
    constexpr int CENTRES = GRID_HEIGHT * GRID_WIDTH, LAG = GRID_WIDTH + 2;
    struct column_t { act_t v[IN_PER_CYCLE][S::COLUMN_VALUES]; };
    struct above_t { act_t v[IN_PER_CYCLE][S::ABOVE_VALUES]; };
    struct row_t { act_t v[IN_PER_CYCLE][S::ROW_VALUES]; };

    column_t slots[4][IN_GROUPS];
    #pragma HLS ARRAY_PARTITION variable=slots type=complete dim=1
    #pragma HLS AGGREGATE variable=slots compact=bit
    above_t line_above[GRID_WIDTH][IN_GROUPS];
    row_t line_row[GRID_WIDTH][IN_GROUPS];
    #pragma HLS AGGREGATE variable=line_above compact=bit
    #pragma HLS AGGREGATE variable=line_row compact=bit
    wino_sum_t accumulators[OUT_PER_CYCLE][4][4];
    #pragma HLS ARRAY_PARTITION variable=accumulators type=complete dim=0
    act_t input[PIXELS_IN * IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=input type=complete
    act_t outputs[4 * OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=outputs type=complete

    // Only taps the centre's position zeroes read these before they are written.
    clear: for (int k = 0; k < GRID_WIDTH * IN_GROUPS; k++) {
        #pragma HLS PIPELINE II=1
        above_t above;
        row_t row;
        for (int i = 0; i < IN_PER_CYCLE; i++) {
            #pragma HLS UNROLL
            for (int q = 0; q < S::ABOVE_VALUES; q++) above.v[i][q] = 0;
            for (int q = 0; q < S::ROW_VALUES; q++) row.v[i][q] = 0;
        }
        line_above[k / IN_GROUPS][k % IN_GROUPS] = above;
        line_row[k / IN_GROUPS][k % IN_GROUPS] = row;
    }

    const int centres = patches * CENTRES;
    int period = 0;                // input `period` comes in, centre period - LAG is done
    int out_group = 0, in_group = 0;
    int in_col = 0;
    int fill_slot = (4 - GRID_WIDTH % 4) % 4;   // (period - GRID_WIDTH) mod 4
    int centre_row = 0, centre_col = 0;

    constexpr int TRIPS = (CENTRES + LAG) * IN_GROUPS * OUT_GROUPS;
    constexpr int MAX_TRIPS = (665 * CENTRES + LAG) * IN_GROUPS * OUT_GROUPS;
    (void)TRIPS; (void)MAX_TRIPS;   // only the pragma uses them
    conv: for (int t = 0; t < (centres + LAG) * IN_GROUPS * OUT_GROUPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=TRIPS max=MAX_TRIPS
        // A slot is written a period before it is read; a line buffer column is
        // read, then written, once a period.
        #pragma HLS DEPENDENCE variable=slots type=inter false
        #pragma HLS DEPENDENCE variable=line_above type=inter false
        #pragma HLS DEPENDENCE variable=line_row type=inter false
        const bool fill = out_group == 0;   // the first IN_GROUPS steps fill the next column
        const bool live = period >= LAG && period < centres + LAG;

        if (fill && in_group == 0) {
            const bool have = period < centres;
            vec_t<act_t, PIXELS_IN * IN_CHANNELS> value;
            if (have) value = in.read();
            for (int i = 0; i < PIXELS_IN * IN_CHANNELS; i++) {
                #pragma HLS UNROLL
                input[i] = have ? value.v[i] : (act_t)0;
            }
        }

        if (fill) {
            // The rows above come from the line buffers, the rows below from the input.
            const above_t above = line_above[in_col][in_group];
            const row_t row = line_row[in_col][in_group];
            column_t column;
            above_t next_above;
            row_t next_row;
            for (int i = 0; i < IN_PER_CYCLE; i++) {
                #pragma HLS UNROLL
                const int ch = in_group * IN_PER_CYCLE + i;
                if (UP) {
                    column.v[i][0] = above.v[i][0];
                    column.v[i][1] = row.v[i][0];
                    column.v[i][2] = input[ch];
                    next_above.v[i][0] = row.v[i][0];
                    next_row.v[i][0] = input[ch];
                } else {
                    // [tile row][dx]: the pixel row above, the block, the row below.
                    for (int dx = 0; dx < 2; dx++) {
                        #pragma HLS UNROLL
                        column.v[i][dx] = above.v[i][dx];
                        column.v[i][2 + dx] = row.v[i][dx];
                        column.v[i][4 + dx] = row.v[i][2 + dx];
                        column.v[i][6 + dx] = input[dx * IN_CHANNELS + ch];
                        next_above.v[i][dx] = row.v[i][2 + dx];
                    }
                    for (int q = 0; q < 4; q++) {
                        #pragma HLS UNROLL
                        next_row.v[i][q] = input[q * IN_CHANNELS + ch];
                    }
                }
            }
            line_above[in_col][in_group] = next_above;
            line_row[in_col][in_group] = next_row;
            for (int k = 0; k < 4; k++) {
                #pragma HLS UNROLL
                if (fill_slot == k) slots[k][in_group] = column;
            }
        }

        column_t columns[4];
        #pragma HLS ARRAY_PARTITION variable=columns type=complete
        for (int k = 0; k < 4; k++) {
            #pragma HLS UNROLL
            columns[k] = slots[k][in_group];
        }
        const column_t &left = columns[(fill_slot + 1) & 3];
        const column_t &centre = columns[(fill_slot + 2) & 3];
        const column_t &right = columns[(fill_slot + 3) & 3];
        const bool has_above = centre_row > 0, has_below = centre_row < GRID_HEIGHT - 1;
        const bool has_left = centre_col > 0, has_right = centre_col < GRID_WIDTH - 1;

        wino_input_t v[IN_PER_CYCLE][4][4];
        #pragma HLS ARRAY_PARTITION variable=v type=complete dim=0
        for (int i = 0; i < IN_PER_CYCLE; i++) {
            #pragma HLS UNROLL
            if (UP) {
                act_t neighbourhood[3][3];
                for (int r = 0; r < 3; r++) {
                    #pragma HLS UNROLL
                    const bool row_inside = r == 0 ? has_above : r == 2 ? has_below : true;
                    neighbourhood[r][0] = row_inside && has_left ? left.v[i][r] : (act_t)0;
                    neighbourhood[r][1] = row_inside ? centre.v[i][r] : (act_t)0;
                    neighbourhood[r][2] = row_inside && has_right ? right.v[i][r] : (act_t)0;
                }
                wino_in_up(neighbourhood, v[i]);
            } else {
                act_t tile[4][4];
                for (int r = 0; r < 4; r++) {
                    #pragma HLS UNROLL
                    const bool row_inside = r == 0 ? has_above : r == 3 ? has_below : true;
                    tile[r][0] = row_inside && has_left ? left.v[i][2 * r + 1] : (act_t)0;
                    tile[r][1] = row_inside ? centre.v[i][2 * r] : (act_t)0;
                    tile[r][2] = row_inside ? centre.v[i][2 * r + 1] : (act_t)0;
                    tile[r][3] = row_inside && has_right ? right.v[i][2 * r] : (act_t)0;
                }
                wino_in_tile(tile, v[i]);
            }
        }

        const wino_word_t<S::WEIGHTS_PER_STEP> word = weight[out_group * IN_GROUPS + in_group];
        for (int j = 0; j < OUT_PER_CYCLE; j++) {
            #pragma HLS UNROLL
            for (int k = 0; k < PRODUCTS; k++) {
                #pragma HLS UNROLL
                const int r = wino_row(UP, k), c = wino_col(UP, k);
                wino_sum_t sum = 0;
                for (int i = 0; i < IN_PER_CYCLE; i++) {
                    #pragma HLS UNROLL
                    sum += mul<DSP, wino_product_t>(v[i][r][c], word.v[(i * OUT_PER_CYCLE + j) * PRODUCTS + k]);
                }
                accumulators[j][r][c] = (in_group == 0 ? (wino_sum_t)0 : accumulators[j][r][c]) + sum;
            }
            if (UP) {
                for (int k = 0; k < 4; k++) {
                    #pragma HLS UNROLL
                    accumulators[j][2][k] = 0;
                    accumulators[j][k][2] = 0;
                }
            }
            if (in_group == IN_GROUPS - 1) {
                wino_sum_t y[2][2];
                wino_out(accumulators[j], y);
                for (int q = 0; q < 4; q++) {
                    #pragma HLS UNROLL
                    const acc_t sum = (acc_t)(y[q / 2][q % 2] >> 2) + bias[out_group][j];
                    for (int og = 0; og < OUT_GROUPS; og++) {
                        #pragma HLS UNROLL
                        if (out_group == og) outputs[q * OUT_CHANNELS + og * OUT_PER_CYCLE + j] = requant_relu<SHIFT>(sum);
                    }
                }
            }
        }
        if (live && out_group == OUT_GROUPS - 1 && in_group == IN_GROUPS - 1) {
            vec_t<act_t, 4 * OUT_CHANNELS> block;
            for (int i = 0; i < 4 * OUT_CHANNELS; i++) {
                #pragma HLS UNROLL
                block.v[i] = outputs[i];
            }
            out.write(block);
        }

        if (in_group < IN_GROUPS - 1) {
            in_group++;
        } else {
            in_group = 0;
            if (out_group < OUT_GROUPS - 1) {
                out_group++;
            } else {
                out_group = 0;
                if (live) {
                    if (centre_col < GRID_WIDTH - 1) centre_col++;
                    else { centre_col = 0; centre_row = centre_row < GRID_HEIGHT - 1 ? centre_row + 1 : 0; }
                }
                in_col = in_col < GRID_WIDTH - 1 ? in_col + 1 : 0;
                fill_slot = (fill_slot + 1) & 3;
                period++;
            }
        }
    }
}

// The weight memory is block RAM when deep, LUTs when shallow. impl takes only a
// literal word: one branch each.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int GRID_HEIGHT, int GRID_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int OUT_PER_CYCLE, bool UP, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS], bool DSP>
void wino_layer(
    hls::stream<vec_t<act_t, wino_shape<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP>::PIXELS_IN
                             * IN_CHANNELS> >& in,
    hls::stream<vec_t<act_t, 4 * OUT_CHANNELS> >& out, int height, int width) {
    #pragma HLS INLINE
    typedef wino_shape<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP> S;
    acc_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    if (S::STEPS > 64) {
        wino_word_t<S::WEIGHTS_PER_STEP> weight[S::STEPS];
        #pragma HLS AGGREGATE variable=weight compact=bit
        #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=bram
        load_wino_weights<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP, BIAS_SHIFT, WEIGHT, BIAS>(
            weight, bias);
        wino_run<GRID_HEIGHT, GRID_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP, SHIFT, DSP>(
            in, weight, bias, out, patches);
    } else {
        wino_word_t<S::WEIGHTS_PER_STEP> weight[S::STEPS];
        #pragma HLS AGGREGATE variable=weight compact=bit
        #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=lutram
        load_wino_weights<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP, BIAS_SHIFT, WEIGHT, BIAS>(
            weight, bias);
        wino_run<GRID_HEIGHT, GRID_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, UP, SHIFT, DSP>(
            in, weight, bias, out, patches);
    }
}

// A 3x3 conv and its ReLU on a map in block order: blocks in, blocks out.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int OUT_PER_CYCLE, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS], bool DSP>
void wino_conv3x3(hls::stream<vec_t<act_t, 4 * IN_CHANNELS> >& in, hls::stream<vec_t<act_t, 4 * OUT_CHANNELS> >& out,
                  int height, int width) {
    wino_layer<PATCH_HEIGHT, PATCH_WIDTH, MAP_HEIGHT / 2, MAP_WIDTH / 2, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE,
               OUT_PER_CYCLE, false, BIAS_SHIFT, SHIFT, WEIGHT, BIAS, DSP>(in, out, height, width);
}

// An UpBlock and its ReLU: a low-res map in raster order, one pixel a vector;
// out one block of the upsampled map per input pixel.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int OUT_PER_CYCLE, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS], bool DSP>
void wino_up_conv3x3(hls::stream<vec_t<act_t, IN_CHANNELS> >& in, hls::stream<vec_t<act_t, 4 * OUT_CHANNELS> >& out,
                     int height, int width) {
    wino_layer<PATCH_HEIGHT, PATCH_WIDTH, MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE,
               OUT_PER_CYCLE, true, BIAS_SHIFT, SHIFT, WEIGHT, BIAS, DSP>(in, out, height, width);
}
