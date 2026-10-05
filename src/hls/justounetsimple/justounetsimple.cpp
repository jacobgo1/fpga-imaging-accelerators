#include "justounetsimple.hpp"
#include <type_traits>

// The quantized weights, read with the header's int8_t meaning float: the
// header declares its arrays inside this namespace, so its unqualified int8_t
// finds this alias first. The ROM then holds the same integers as exact floats,
// and the hardware multiplies float by float, with no int-to-float converter.
// (Vitis 2025.2 builds such a converter for int8 weights and leaves its module
// out of the exported IP, which fails Vivado synthesis.)
namespace justounetsimple_int8 { using int8_t = float; }
#include "../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"

namespace q = justounetsimple_int8;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                  decltype(q::conv1_weight)>::type>::type, float>::value,
              "the weights must be read as float; did the header's element type change?");

// Each tensor's 2^-frac, computed by the compiler.
constexpr float W1_SCALE = pow2_neg(q::conv1_weight_frac), B1_SCALE = pow2_neg(q::conv1_bias_frac);
constexpr float W2_SCALE = pow2_neg(q::conv2_weight_frac), B2_SCALE = pow2_neg(q::conv2_bias_frac);
constexpr float W3_SCALE = pow2_neg(q::conv3_weight_frac), B3_SCALE = pow2_neg(q::conv3_bias_frac);
constexpr float W4_SCALE = pow2_neg(q::conv4_weight_frac), B4_SCALE = pow2_neg(q::conv4_bias_frac);

void justounetsimple(data_t din[JUNETS_H][JUNETS_W][JUNETS_IN_CH],
                     data_t dout[JUNETS_H][JUNETS_W][JUNETS_OUT_CH])
{
    // The patch in and the scores out go through one AXI master; the PS sets
    // their DDR addresses in the s_axilite registers before pulsing start.
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=s_axilite port=din    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Read the patch from DDR once. Otherwise every multiply in the
    // convolution would be a separate DDR access.
    static data_t patch[JUNETS_H][JUNETS_W][JUNETS_IN_CH];
    for (int i = 0; i < JUNETS_H; i++)
        for (int j = 0; j < JUNETS_W; j++)
            for (int c = 0; c < JUNETS_IN_CH; c++)
                patch[i][j][c] = din[i][j][c];

    static data_t scores[JUNETS_H][JUNETS_W][JUNETS_OUT_CH];
    justounetsimple_golden<data_t, float, JUNETS_H, JUNETS_W, JUNETS_IN_CH,
                           JUNETS_BASE_CH, JUNETS_OUT_CH, JUNETS_K>(
        patch,
        q::conv1_weight, W1_SCALE, q::conv1_bias, B1_SCALE,
        q::conv2_weight, W2_SCALE, q::conv2_bias, B2_SCALE,
        q::conv3_weight, W3_SCALE, q::conv3_bias, B3_SCALE,
        q::conv4_weight, W4_SCALE, q::conv4_bias, B4_SCALE,
        scores);

    for (int i = 0; i < JUNETS_H; i++)
        for (int j = 0; j < JUNETS_W; j++)
            for (int c = 0; c < JUNETS_OUT_CH; c++)
                dout[i][j][c] = scores[i][j][c];
}
