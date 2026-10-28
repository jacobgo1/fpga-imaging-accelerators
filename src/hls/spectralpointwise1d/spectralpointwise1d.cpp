#include "spectralpointwise1d.hpp"
#include <type_traits>

// The quantized weights, read with the header's int8_t meaning float: the
// header declares its arrays inside this namespace, so its unqualified int8_t
// finds this alias first. The ROM then holds the same integers as exact floats,
// and the hardware multiplies float by float, with no int-to-float converter
// (as in justoliunet.cpp: Vitis 2025.2 leaves that converter's module out of
// the exported IP).
namespace spectralpointwise1d_int8 { using int8_t = float; }
#include "../../../weights/quantized/spectralpointwise1d/spectralpointwise1d_int8.hpp"

namespace q = spectralpointwise1d_int8;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                  decltype(q::convsp1_weight)>::type>::type, float>::value,
              "the weights must be read as float; did the header's element type change?");

// Each tensor's 2^-frac, computed by the compiler.
constexpr float W1_SCALE = pow2_neg(q::convsp1_weight_frac), B1_SCALE = pow2_neg(q::convsp1_bias_frac);
constexpr float W2_SCALE = pow2_neg(q::convsp2_weight_frac), B2_SCALE = pow2_neg(q::convsp2_bias_frac);
constexpr float W3_SCALE = pow2_neg(q::convsp3_weight_frac), B3_SCALE = pow2_neg(q::convsp3_bias_frac);

void spectralpointwise1d(const data_t *din, data_t *dout, int n)
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
        // convsp1 would be a separate DDR access.
        data_t spectrum[SPW_BANDS];
        read: for (int b = 0; b < SPW_BANDS; b++)
            spectrum[b] = din[p * SPW_BANDS + b];

        data_t scores[SPW_CLASSES];
        spectralpointwise1d_golden<data_t, float, SPW_BANDS, SPW_CH1, SPW_CH2, SPW_CLASSES>(
            spectrum,
            q::convsp1_weight, W1_SCALE, q::convsp1_bias, B1_SCALE,
            q::convsp2_weight, W2_SCALE, q::convsp2_bias, B2_SCALE,
            q::convsp3_weight, W3_SCALE, q::convsp3_bias, B3_SCALE,
            scores);

        write: for (int c = 0; c < SPW_CLASSES; c++)
            dout[p * SPW_CLASSES + c] = scores[c];
    }
}
