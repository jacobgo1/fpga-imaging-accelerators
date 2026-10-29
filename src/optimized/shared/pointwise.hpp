#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// A 1x1 conv on int16 activations: IN_PER_CYCLE input channels times OUT_PER_CYCLE
// output channels a cycle, so a layer with time to spare uses fewer multipliers.
// RELU: outputs ReLU'd and rounded off SHIFT bits to activations; otherwise the
// sums, for the scores. PIXELS_PER_WORD = 4 sends each 2x2 block's pixels
// together, for a next layer in block order.
template<bool RELU, int SHIFT> struct pointwise_out {
    typedef act_t type;
    static type of(acc_t sum) {
        #pragma HLS INLINE
        return requant_relu<SHIFT>(sum);
    }
};
template<int SHIFT> struct pointwise_out<false, SHIFT> {
    typedef acc_t type;
    static type of(acc_t sum) {
        #pragma HLS INLINE
        return sum;
    }
};

template<int PATCH_HEIGHT, int PATCH_WIDTH, int PIXELS, int IN_CHANNELS, int OUT_CHANNELS, int OUT_PER_CYCLE,
         int PIXELS_PER_WORD, bool RELU, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][1][1], const wgt_t (&BIAS)[OUT_CHANNELS], bool DSP,
         int IN_PER_CYCLE = IN_CHANNELS>
void pointwise(hls::stream<vec_t<act_t, IN_CHANNELS> >& in,
               hls::stream<vec_t<typename pointwise_out<RELU, SHIFT>::type, PIXELS_PER_WORD * OUT_CHANNELS> >& out,
               int height, int width) {
    typedef typename pointwise_out<RELU, SHIFT>::type out_t;
    constexpr int OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE, IN_GROUPS = IN_CHANNELS / IN_PER_CYCLE;
    constexpr int STEPS = OUT_GROUPS * IN_GROUPS;   // per pixel: output group outer, input group inner
    wgt_t weight[STEPS][OUT_PER_CYCLE][IN_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=3
    acc_t bias[OUT_GROUPS][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    load: for (int o = 0; o < OUT_CHANNELS; o++) {
        for (int i = 0; i < IN_CHANNELS; i++) {
            #pragma HLS PIPELINE II=1
            weight[(o / OUT_PER_CYCLE) * IN_GROUPS + i / IN_PER_CYCLE][o % OUT_PER_CYCLE][i % IN_PER_CYCLE] =
                WEIGHT[o][i][0][0];
        }
        bias[o / OUT_PER_CYCLE][o % OUT_PER_CYCLE] = (acc_t)BIAS[o] * ((acc_t)1 << BIAS_SHIFT);
    }

    act_t pixel[IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=pixel type=complete
    acc_t running[OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=running type=complete
    out_t outputs[PIXELS_PER_WORD * OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=outputs type=complete
    const int pixels = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * PIXELS;
    int out_group = 0, in_group = 0, word_pixel = 0;
    constexpr int TRIPS = PIXELS * STEPS, MAX_TRIPS = 665 * TRIPS;
    (void)TRIPS; (void)MAX_TRIPS;   // only the pragma uses them
    px: for (int t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=TRIPS max=MAX_TRIPS
        if (out_group == 0 && in_group == 0) {
            const vec_t<act_t, IN_CHANNELS> value = in.read();
            for (int i = 0; i < IN_CHANNELS; i++) {
                #pragma HLS UNROLL
                pixel[i] = value.v[i];
            }
        }
        const int step = out_group * IN_GROUPS + in_group;
        for (int j = 0; j < OUT_PER_CYCLE; j++) {
            #pragma HLS UNROLL
            acc_t sum = in_group == 0 ? bias[out_group][j] : running[j];
            for (int i = 0; i < IN_PER_CYCLE; i++) {
                #pragma HLS UNROLL
                sum += mul<DSP, acc_t>(pixel[(IN_GROUPS == 1 ? 0 : in_group * IN_PER_CYCLE) + i], weight[step][j][i]);
            }
            running[j] = sum;
            const out_t result = pointwise_out<RELU, SHIFT>::of(sum);
            // Written at fixed places, the run-time choice in which statement runs.
            for (int og = 0; og < OUT_GROUPS; og++) {
                #pragma HLS UNROLL
                for (int p = 0; p < PIXELS_PER_WORD; p++) {
                    #pragma HLS UNROLL
                    if (in_group == IN_GROUPS - 1 && out_group == og && word_pixel == p)
                        outputs[p * OUT_CHANNELS + og * OUT_PER_CYCLE + j] = result;
                }
            }
        }
        if (out_group == OUT_GROUPS - 1 && in_group == IN_GROUPS - 1 && word_pixel == PIXELS_PER_WORD - 1) {
            vec_t<out_t, PIXELS_PER_WORD * OUT_CHANNELS> word;
            for (int i = 0; i < PIXELS_PER_WORD * OUT_CHANNELS; i++) {
                #pragma HLS UNROLL
                word.v[i] = outputs[i];
            }
            out.write(word);
        }
        if (in_group < IN_GROUPS - 1) {
            in_group++;
        } else {
            in_group = 0;
            if (out_group < OUT_GROUPS - 1) {
                out_group++;
            } else {
                out_group = 0;
                word_pixel = word_pixel < PIXELS_PER_WORD - 1 ? word_pixel + 1 : 0;
            }
        }
    }
}
