// Testbench of the read-only kernel: the checksum must be every band of every pixel
// of every patch, the bottom row and right column repeated as the readers do, and
// the kernel must write nothing past it.
#include "justounetsimple_opt_readonly.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;

static void expected_checksum(const std::vector<cube_word_t>& cube, int height, int width,
                              uint16_t band_sums[BANDS_PER_WORD]) {
    for (int band = 0; band < BANDS_PER_WORD; band++) band_sums[band] = 0;
    const int padded_height = (height + PATCH_SIZE - 1) / PATCH_SIZE * PATCH_SIZE;
    const int padded_width = (width + PATCH_SIZE - 1) / PATCH_SIZE * PATCH_SIZE;
    for (int y = 0; y < padded_height; y++)
        for (int x = 0; x < padded_width; x++) {
            const int pixel = std::min(y, height - 1) * width + std::min(x, width - 1);
            for (int word = 0; word < WORDS_PER_PIXEL; word++)
                for (int band = 0; band < BANDS_PER_WORD; band++)
                    band_sums[band] += cube[pixel * WORDS_PER_PIXEL + word][band];
        }
}

static bool check(int height, int width, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> random_value(0, 65535);
    std::vector<cube_word_t> cube(height * width * WORDS_PER_PIXEL);
    for (auto& word : cube)
        for (int band = 0; band < BANDS_PER_WORD; band++) word[band] = (uint16_t)random_value(rng);

    const int32_t UNTOUCHED = 0x5a5a5a5a;
    std::vector<score_word_t> dout(2);
    for (auto& word : dout)
        for (int i = 0; i < 4; i++) word[i] = UNTOUCHED;

    cube_word_t *d = cube.data();
    uint32_t cycles = 0;   // only counts in co-simulation
    justounetsimple_opt_readonly(d, d, d, d, dout.data(), height, width, cycles);

    uint16_t want[BANDS_PER_WORD];
    expected_checksum(cube, height, width, want);
    bool ok = true;
    for (int band = 0; band < BANDS_PER_WORD; band++) {
        const uint16_t got = (uint16_t)((uint32_t)dout[0][band / 2] >> (16 * (band % 2)));
        if (got != want[band]) {
            std::printf("  %d x %d: band %d sums to %u, expected %u\n", height, width, band, got, want[band]);
            ok = false;
        }
    }
    for (int i = 0; i < 4; i++)
        if (dout[1][i] != UNTOUCHED) {
            std::printf("  %d x %d: wrote past the checksum\n", height, width);
            ok = false;
            break;
        }
    std::printf("%s %d x %d, cycles register: %u\n", ok ? "PASS" : "FAIL", height, width, (unsigned)cycles);
    return ok;
}

int main() {
    // 40 x 50 matches the m_axi depth pragmas; then one whole patch, one pixel, a strip.
    const int sizes[][2] = {{40, 50}, {32, 32}, {1, 1}, {7, 70}};
    bool ok = true;
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
        ok = check(sizes[i][0], sizes[i][1], 11 + i) && ok;
    std::printf(ok ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    return ok ? 0 : 1;
}
