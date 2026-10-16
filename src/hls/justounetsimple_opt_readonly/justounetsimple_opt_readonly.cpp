#include "justounetsimple_opt_readonly.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/shared/count_cycles.hpp"

constexpr int WORDS = RAW_BANDS / PORT_BANDS;        // 15 words per pixel in the cube
static_assert(RAW_BANDS % PORT_BANDS == 0, "a pixel is whole words");

// drain: in patch_buffer's place, a word from every reader each cycle. Each
// reader sends PH / PORTS rows x PW pixels x WORDS words a patch (the edge
// patches padded to full size), so drain takes exactly that many: fewer would
// leave the readers stuck on full FIFOs, more would wait forever. The sum
// keeps the reads from being optimized away. Then a word on finished, for
// count_cycles, as store_scores does in justounetsimple_opt.
template<int PH, int PW, int WORDS, int PORTS>
void drain(hls::stream<jopt_raw_t> in[PORTS], hls::burst_maxi<jopt_out_t> dout, int height, int width,
           hls::stream<bool>& finished) {
    static_assert(PORT_BANDS % 2 == 0, "two lanes per int32 of dout");
    constexpr int PER_PATCH = PH / PORTS * PW * WORDS;
    const int patches = patch_count<PH, PW>(height, width);
    uint16_t sum[PORT_BANDS];
    #pragma HLS ARRAY_PARTITION variable=sum type=complete
    for (int b = 0; b < PORT_BANDS; b++) sum[b] = 0;
    patch: for (int p = 0; p < patches; p++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        word: for (int k = 0; k < PER_PATCH; k++) {
            #pragma HLS PIPELINE II=1
            jopt_raw_t v[PORTS];
            for (int i = 0; i < PORTS; i++) {
                #pragma HLS UNROLL
                v[i] = in[i].read();
            }
            for (int b = 0; b < PORT_BANDS; b++) {
                #pragma HLS UNROLL
                uint16_t s = sum[b];
                for (int i = 0; i < PORTS; i++) {
                    #pragma HLS UNROLL
                    s += v[i][b];
                }
                sum[b] = s;
            }
        }
    }
    jopt_out_t out;
    for (int i = 0; i < PORT_BANDS / 2; i++) {
        #pragma HLS UNROLL
        out[i] = (int32_t)((uint32_t)sum[2 * i] | ((uint32_t)sum[2 * i + 1] << 16));
    }
    dout.write_request(0, 1);
    dout.write(out);
    dout.write_response();
    finished.write(true);
}

void justounetsimple_opt_readonly(hls::burst_maxi<jopt_raw_t> din0, hls::burst_maxi<jopt_raw_t> din1,
                                  hls::burst_maxi<jopt_raw_t> din2, hls::burst_maxi<jopt_raw_t> din3,
                                  hls::burst_maxi<jopt_out_t> dout, int height, int width,
                                  uint32_t& cycles)
{
// The interfaces of justounetsimple_opt, unchanged, so the reads are the same
// and the same block design (boards/zcu104/system.tcl) wires it the same way.
// depth is only for co-simulation: the testbench's 40 x 50 image.
#pragma HLS INTERFACE mode = m_axi port = din0 bundle = gmem0 offset = slave depth = 30000 max_read_burst_length = 256 num_read_outstanding = 4
#pragma HLS INTERFACE mode = m_axi port = din1 bundle = gmem1 offset = slave depth = 30000 max_read_burst_length = 256 num_read_outstanding = 4
#pragma HLS INTERFACE mode = m_axi port = din2 bundle = gmem2 offset = slave depth = 30000 max_read_burst_length = 256 num_read_outstanding = 4
#pragma HLS INTERFACE mode = m_axi port = din3 bundle = gmem3 offset = slave depth = 30000 max_read_burst_length = 256 num_read_outstanding = 4
#pragma HLS INTERFACE mode = m_axi port = dout bundle = gmem4 offset = slave depth = 2000 max_write_burst_length = 32 num_write_outstanding = 16
#pragma HLS INTERFACE mode = s_axilite port = din0 bundle = control
#pragma HLS INTERFACE mode = s_axilite port = din1 bundle = control
#pragma HLS INTERFACE mode = s_axilite port = din2 bundle = control
#pragma HLS INTERFACE mode = s_axilite port = din3 bundle = control
#pragma HLS INTERFACE mode = s_axilite port = dout bundle = control
#pragma HLS INTERFACE mode = s_axilite port = height bundle = control
#pragma HLS INTERFACE mode = s_axilite port = width bundle = control
#pragma HLS INTERFACE mode = s_axilite port = cycles bundle = control
#pragma HLS INTERFACE mode = s_axilite port = return bundle = control
#pragma HLS DATAFLOW

    hls::stream<jopt_raw_t> s_raw[PORTS];
    hls::stream<bool> s_finished("s_finished");
// As in justounetsimple_opt: the readers run ahead of whoever takes their words.
#pragma HLS STREAM variable = s_raw depth = 64

    read_rows_ahead<H, W, WORDS, PORTS, 0>(din0, s_raw[0], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 1>(din1, s_raw[1], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 2>(din2, s_raw[2], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 3>(din3, s_raw[3], height, width);
    drain<H, W, WORDS, PORTS>(s_raw, dout, height, width, s_finished);
    // Last: C simulation runs the processes in this order (count_cycles.hpp).
    count_cycles(s_finished, cycles);
}
