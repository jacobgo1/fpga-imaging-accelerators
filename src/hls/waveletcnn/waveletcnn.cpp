#include "waveletcnn.hpp"
#include <type_traits>

// The quantized weights, read with the header's int8_t meaning float: the
// header declares its arrays inside this namespace, so its unqualified int8_t
// finds this alias first. The ROM then holds the same integers as exact floats,
// and the hardware multiplies float by float, with no int-to-float converter.
// (Vitis 2025.2 builds such a converter for int8 weights and leaves its module
// out of the exported IP, which fails Vivado synthesis.)
namespace waveletcnn_int8 { using int8_t = float; }
#include "../../../weights/quantized/waveletcnn/waveletcnn_int8.hpp"

namespace q = waveletcnn_int8;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                  decltype(q::projection_weight)>::type>::type, float>::value,
              "the weights must be read as float; did the header's element type change?");

// Each tensor's 2^-frac, computed by the compiler.
constexpr float H0_SCALE = pow2_neg(q::dwt_spectral_h0_frac);
constexpr float H1_SCALE = pow2_neg(q::dwt_spectral_h1_frac);
constexpr float PW_SCALE = pow2_neg(q::projection_weight_frac),    PB_SCALE = pow2_neg(q::projection_bias_frac);
constexpr float CW_SCALE = pow2_neg(q::conv_block_0_weight_frac),  CB_SCALE = pow2_neg(q::conv_block_0_bias_frac);
constexpr float HW_SCALE = pow2_neg(q::seg_head_0_weight_frac),    HB_SCALE = pow2_neg(q::seg_head_0_bias_frac);

void waveletcnn(data_t din[WCNN_H][WCNN_W][WCNN_IN_CH],
                data_t dout[WCNN_H][WCNN_W][WCNN_OUT_CH])
{
    // The patch in and the scores out go through one AXI master; the PS sets
    // their DDR addresses in the s_axilite registers before pulsing start.
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=s_axilite port=din    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Read the patch from DDR once. Otherwise every multiply in the DWT and
    // projection would be a separate DDR access.
    static data_t patch[WCNN_H][WCNN_W][WCNN_IN_CH];
    for (int i = 0; i < WCNN_H; i++)
        for (int j = 0; j < WCNN_W; j++)
            for (int c = 0; c < WCNN_IN_CH; c++)
                patch[i][j][c] = din[i][j][c];

    static data_t scores[WCNN_H][WCNN_W][WCNN_OUT_CH];
    waveletcnn_golden<data_t, float, WCNN_H, WCNN_W, WCNN_IN_CH,
                      WCNN_MID_CH, WCNN_OUT_CH, WCNN_DWT_LEN>(
        patch,
        q::dwt_spectral_h0, H0_SCALE, q::dwt_spectral_h1, H1_SCALE,
        q::projection_weight, PW_SCALE, q::projection_bias, PB_SCALE,
        q::conv_block_0_weight, CW_SCALE, q::conv_block_0_bias, CB_SCALE,
        q::seg_head_0_weight, HW_SCALE, q::seg_head_0_bias, HB_SCALE,
        scores);

    for (int i = 0; i < WCNN_H; i++)
        for (int j = 0; j < WCNN_W; j++)
            for (int c = 0; c < WCNN_OUT_CH; c++)
                dout[i][j][c] = scores[i][j][c];
}
