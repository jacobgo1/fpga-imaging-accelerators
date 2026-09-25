#include "justoliunet.hpp"

void justoliunet(
    data_t din[JNET_H][JNET_W][1],
    data_t w1[JNET_BASE_CH][1][JNET_K],                    data_t b1[JNET_BASE_CH],
    data_t w2[2 * JNET_BASE_CH][JNET_BASE_CH][JNET_K],      data_t b2[2 * JNET_BASE_CH],
    data_t w3[3 * JNET_BASE_CH][2 * JNET_BASE_CH][JNET_K],  data_t b3[3 * JNET_BASE_CH],
    data_t w4[4 * JNET_BASE_CH][3 * JNET_BASE_CH][JNET_K],  data_t b4[4 * JNET_BASE_CH],
    data_t w5[JNET_NUM_CLASSES][JNET_FLAT],                 data_t b5[JNET_NUM_CLASSES],
    data_t dout[JNET_H][JNET_NUM_CLASSES])
{
    // One shared AXI master (bundle=gmem) rather than one per array: this
    // kernel DMAs every input/output out of the same DDR region the PS
    // filled, instead of exposing 12 separate BRAM ports that would each
    // need their own controller wired by hand in the block design (see
    // software/zcu104/ and boards/README.md for the PS side of this).
    // Every array also gets an s_axilite entry: that is what turns its
    // pointer into a settable register, so the PS can point each argument
    // at wherever in the DDR buffer it put that array before pulsing start.
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=w1   bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=b1   bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=w2   bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=b2   bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=w3   bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=b3   bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=w4   bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=b4   bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=w5   bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=b5   bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem offset=slave

    #pragma HLS INTERFACE mode=s_axilite port=din  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=w1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=b1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=w2   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=b2   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=w3   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=b3   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=w4   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=b4   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=w5   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=b5   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the golden reference and the
    // native/A53 testbench both make (see justoliunet.hpp part 1). No
    // staging buffers, no partitioning, no directives -- that is the
    // baseline this kernel exists to measure.
    justoliunet_golden<data_t, JNET_H, JNET_W, JNET_K, JNET_BASE_CH, JNET_NUM_CLASSES>(
        din, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, dout);
}
