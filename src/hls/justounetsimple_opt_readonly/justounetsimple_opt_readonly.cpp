#include "justounetsimple_opt_readonly.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/shared/count_cycles.hpp"

constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;

// Takes a word from every reader each cycle, as patch_buffer does, and adds
// them up so HLS keeps every read. It must take exactly as many words as the
// readers send, or they stall.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int WORDS_PER_PIXEL, int READ_PORTS>
void sum_all_words(hls::stream<cube_word_t> cube_words[READ_PORTS], hls::burst_maxi<score_word_t> dout,
                   int height, int width, hls::stream<bool>& all_written)
{
    constexpr int WORDS_PER_PORT_AND_PATCH = PATCH_HEIGHT / READ_PORTS * PATCH_WIDTH * WORDS_PER_PIXEL;
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);

    uint16_t band_sums[BANDS_PER_WORD];
    #pragma HLS ARRAY_PARTITION variable=band_sums type=complete
    for (int band = 0; band < BANDS_PER_WORD; band++)
        band_sums[band] = 0;

    patch: for (int p = 0; p < patches; p++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        word: for (int w = 0; w < WORDS_PER_PORT_AND_PATCH; w++) {
            #pragma HLS PIPELINE II=1
            cube_word_t words[READ_PORTS];
            for (int port = 0; port < READ_PORTS; port++) {
                #pragma HLS UNROLL
                words[port] = cube_words[port].read();
            }
            for (int band = 0; band < BANDS_PER_WORD; band++) {
                #pragma HLS UNROLL
                uint16_t sum = band_sums[band];
                for (int port = 0; port < READ_PORTS; port++) {
                    #pragma HLS UNROLL
                    sum += words[port][band];
                }
                band_sums[band] = sum;
            }
        }
    }

    score_word_t checksum;
    for (int i = 0; i < BANDS_PER_WORD / 2; i++) {
        #pragma HLS UNROLL
        checksum[i] = (int32_t)((uint32_t)band_sums[2 * i] | ((uint32_t)band_sums[2 * i + 1] << 16));
    }
    dout.write_request(0, 1);
    dout.write(checksum);
    dout.write_response();
    all_written.write(true);
}

void justounetsimple_opt_readonly(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                                  hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                                  hls::burst_maxi<score_word_t> dout, int height, int width,
                                  uint32_t& cycles)
{
    // justounetsimple_opt's interfaces, so the reads are the same.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din2 bundle=gmem2 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din3 bundle=gmem3 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem4 offset=slave depth=2000 max_write_burst_length=32 num_write_outstanding=16
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din2   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din3   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=cycles bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW

    hls::stream<cube_word_t> cube_words[READ_PORTS];
    hls::stream<bool> all_written("all_written");
    #pragma HLS STREAM variable=cube_words depth=64

    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 0>(din0, cube_words[0], height, width);
    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 1>(din1, cube_words[1], height, width);
    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 2>(din2, cube_words[2], height, width);
    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 3>(din3, cube_words[3], height, width);
    sum_all_words<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS>(cube_words, dout, height, width, all_written);

    count_cycles(all_written, cycles);   // last: C simulation runs the processes in order
}
