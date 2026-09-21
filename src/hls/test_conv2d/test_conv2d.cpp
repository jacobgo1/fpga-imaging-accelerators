#include "test_conv2d.hpp"

void test_conv2d(const data_t in[3][28][28],
                 const data_t weight[3][3][3][3],
                 result_t out[3][26][26])
{
    #pragma HLS INTERFACE mode=bram port=in
    #pragma HLS INTERFACE mode=bram port=weight
    #pragma HLS INTERFACE mode=bram port=out
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // ---- local buffers ----
    data_t w_buf[3][3][3][3];
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=0   // all 81 weights in registers

    data_t in_buf[3][28][28];
    #pragma HLS ARRAY_PARTITION variable=in_buf complete dim=1  // 3 input channels in parallel
    #pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=3 dim=2  // rows oh, oh+1, oh+2
    #pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=3 dim=3  // cols ow, ow+1, ow+2

    load_w:
    for (int oc = 0; oc < 3; oc++)
        for (int ic = 0; ic < 3; ic++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++) {
                    #pragma HLS PIPELINE II=1
                    w_buf[oc][ic][kh][kw] = weight[oc][ic][kh][kw];
                }

    load_in:
    for (int ic = 0; ic < 3; ic++)
        for (int h = 0; h < 28; h++)
            for (int w = 0; w < 28; w++) {
                #pragma HLS PIPELINE II=1
                in_buf[ic][h][w] = in[ic][h][w];
            }

    oc_loop:
    for (int oc = 0; oc < 3; oc++){
        oh_loop:
        for (int oh = 0; oh < 26; oh++){
            ow_loop:
            for (int ow = 0; ow < 26; ow++){
                #pragma HLS PIPELINE II=1
                result_t sum = 0;
                ic_loop:
                for (int ic = 0; ic < 3; ic++){
                    #pragma HLS UNROLL
                    kh_loop:
                    for (int kh = 0; kh < 3; kh++){
                        #pragma HLS UNROLL
                        kw_loop:
                        for (int kw = 0; kw < 3; kw++){
                            #pragma HLS UNROLL
                            sum += static_cast<result_t>(in_buf[ic][oh+kh][ow+kw])
                                 * w_buf[oc][ic][kh][kw];
                        }
                    }
                }
                out[oc][oh][ow] = sum;
            }
        }
    }
}