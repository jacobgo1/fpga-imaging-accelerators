#include "justoliunet.hpp"
#include <type_traits>

// The quantized weights, read with the header's int8_t meaning float: the
// header declares its arrays inside this namespace, so its unqualified int8_t
// finds this alias first. The ROM then holds the same integers as exact floats,
// and the hardware multiplies float by float, with no int-to-float converter
// (as in justounetsimple.cpp: Vitis 2025.2 leaves that converter's module out
// of the exported IP).
namespace justoliunet_int8 { using int8_t = float; }
#include "../../../weights/quantized/justoliunet/justoliunet_int8.hpp"

namespace q = justoliunet_int8;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                  decltype(q::conv1_weight)>::type>::type, float>::value,
              "the weights must be read as float; did the header's element type change?");

// Each tensor's 2^-frac, computed by the compiler.
constexpr float W1_SCALE = pow2_neg(q::conv1_weight_frac), B1_SCALE = pow2_neg(q::conv1_bias_frac);
constexpr float W2_SCALE = pow2_neg(q::conv2_weight_frac), B2_SCALE = pow2_neg(q::conv2_bias_frac);
constexpr float W3_SCALE = pow2_neg(q::conv3_weight_frac), B3_SCALE = pow2_neg(q::conv3_bias_frac);
constexpr float W4_SCALE = pow2_neg(q::conv4_weight_frac), B4_SCALE = pow2_neg(q::conv4_bias_frac);
constexpr float WF_SCALE = pow2_neg(q::fc_weight_frac),    BF_SCALE = pow2_neg(q::fc_bias_frac);

void justoliunet(const data_t *din, data_t *dout, int n)
{
    // The spectra in and the scores out go through one AXI master; the PS sets
    // their DDR addresses and n in the s_axilite registers before pulsing start.
    // depth is only for co-simulation: the testbench's 32 pixels.
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem offset=slave depth=3520
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem offset=slave depth=96
    #pragma HLS INTERFACE mode=s_axilite port=din    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=n      bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    pixels: for (int p = 0; p < n; p++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        // Read the pixel's spectrum from DDR once. Otherwise every multiply in
        // conv1 would be a separate DDR access.
        data_t spectrum[JLN_BANDS][1];
        read: for (int b = 0; b < JLN_BANDS; b++)
            spectrum[b][0] = din[p * JLN_BANDS + b];

        data_t scores[JLN_CLASSES];
        justoliunet_golden<data_t, float, JLN_BANDS, JLN_BASE_CH, JLN_CLASSES, JLN_K>(
            spectrum,
            q::conv1_weight, W1_SCALE, q::conv1_bias, B1_SCALE,
            q::conv2_weight, W2_SCALE, q::conv2_bias, B2_SCALE,
            q::conv3_weight, W3_SCALE, q::conv3_bias, B3_SCALE,
            q::conv4_weight, W4_SCALE, q::conv4_bias, B4_SCALE,
            q::fc_weight, WF_SCALE, q::fc_bias, BF_SCALE,
            scores);

        write: for (int c = 0; c < JLN_CLASSES; c++)
            dout[p * JLN_CLASSES + c] = scores[c];
    }
}
