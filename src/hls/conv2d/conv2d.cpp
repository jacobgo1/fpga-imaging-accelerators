#include "conv2d.hpp"

void conv2d(const data_t in    [CONV_IC][CONV_IH][CONV_IW],
                 const data_t weight[CONV_OC][CONV_IC][CONV_KH][CONV_KW],
                 result_t     out   [CONV_OC][CONV_OH][CONV_OW])
{
    #pragma HLS INTERFACE mode=bram port=in
    #pragma HLS INTERFACE mode=bram port=weight
    #pragma HLS INTERFACE mode=bram port=out
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // ---- local buffers -------------------------------------------------
    data_t w_buf[CONV_OC][CONV_IC][CONV_KH][CONV_KW];
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=0

    data_t in_buf[CONV_IC][CONV_IH][CONV_IW];
    #pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=CONV_IC_PAR dim=1
    #pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=CONV_ROW_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=CONV_COL_BANKS dim=3

    // ---- copy interface BRAMs into the partitioned local buffers -------
    load_w_oc:
    for (int oc = 0; oc < CONV_OC; oc++)
        load_w_ic:
        for (int ic = 0; ic < CONV_IC; ic++)
            load_w_kh:
            for (int kh = 0; kh < CONV_KH; kh++)
                load_w_kw:
                for (int kw = 0; kw < CONV_KW; kw++) {
                    #pragma HLS PIPELINE II=1
                    w_buf[oc][ic][kh][kw] = weight[oc][ic][kh][kw];
                }

    load_in_ic:
    for (int ic = 0; ic < CONV_IC; ic++)
        load_in_h:
        for (int h = 0; h < CONV_IH; h++)
            load_in_w:
            for (int w = 0; w < CONV_IW; w++) {
                #pragma HLS PIPELINE II=1
                in_buf[ic][h][w] = in[ic][h][w];
            }

    // ---- convolution ---------------------------------------------------
    result_t acc[CONV_OW];
    #pragma HLS DEPENDENCE variable=acc type=inter false

    oc_loop:
    for (int oc = 0; oc < CONV_OC; oc++) {
        oh_loop:
        for (int oh = 0; oh < CONV_OH; oh++) {
            ict_loop:
            for (int ic0 = 0; ic0 < CONV_IC; ic0 += CONV_IC_PAR) {
                ow_loop:
                for (int ow = 0; ow < CONV_OW; ow++) {
                    #pragma HLS PIPELINE II=1
                    const int ih0 = oh * CONV_STRIDE;
                    const int iw0 = ow * CONV_STRIDE;

                    result_t sum = (ic0 == 0) ? result_t(0) : acc[ow];

                    for (int icp = 0; icp < CONV_IC_PAR; icp++) {
                        #pragma HLS UNROLL
                        for (int kh = 0; kh < CONV_KH; kh++) {
                            #pragma HLS UNROLL
                            for (int kw = 0; kw < CONV_KW; kw++) {
                                #pragma HLS UNROLL
                                // Multiply at 16x16 and widen afterwards. Casting both
                                // operands to result_t first asks for a 64x64 multiplier
                                // (~16 DSPs each, 27 of them here); the int promotion of
                                // two int16_t operands is exact -- |product| <= 2^30.
                                sum += static_cast<result_t>(
                                           in_buf[ic0 + icp][ih0 + kh][iw0 + kw]
                                         * w_buf[oc][ic0 + icp][kh][kw]);
                            }
                        }
                    }

                    acc[ow] = sum;
                    if (ic0 + CONV_IC_PAR >= CONV_IC)
                        out[oc][oh][ow] = sum;
                }
            }
        }
    }
}