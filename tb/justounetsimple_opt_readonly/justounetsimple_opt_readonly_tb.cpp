// justounetsimple_opt_readonly_tb.cpp  --  testbench for the read-only kernel.
// Random raw cubes of a few sizes, the edge patches partly outside the image:
// the checksum must be every band of every pixel of every 32 x 32 patch, the
// bottom row and right column repeated as read_rows does, added up the plain
// way (lane = band % 8, mod 2^16). The kernel writes one word: the next must
// be untouched. A reader that skipped, repeated or swapped rows, or a drain
// that took too few or too many words, fails here (or hangs in a stream read).
#include "justounetsimple_opt_readonly.hpp"
#include <cstdio>
#include <random>
#include <vector>

constexpr int LANES = PORT_BANDS;     // the checksum's uint16 lanes

static void reference(const std::vector<jopt_raw_t>& cube, int h, int w, uint16_t sum[LANES]) {
    const int words = RAW_BANDS / PORT_BANDS;
    for (int b = 0; b < LANES; b++) sum[b] = 0;
    for (int y = 0; y < (h + H - 1) / H * H; y++)
        for (int x = 0; x < (w + W - 1) / W * W; x++) {
            const int yy = y < h ? y : h - 1, xx = x < w ? x : w - 1;
            for (int k = 0; k < words; k++)
                for (int b = 0; b < LANES; b++)
                    sum[b] = (uint16_t)(sum[b] + cube[(yy * w + xx) * words + k][b]);
        }
}

static bool check(int h, int w, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> value(0, 65535);
    std::vector<jopt_raw_t> cube(h * w * RAW_BANDS / PORT_BANDS);
    for (auto& word : cube)
        for (int b = 0; b < PORT_BANDS; b++) word[b] = (uint16_t)value(rng);
    const int32_t SENTINEL = 0x5a5a5a5a;
    std::vector<jopt_out_t> dout(2);
    for (auto& word : dout)
        for (int i = 0; i < 4; i++) word[i] = SENTINEL;

    jopt_raw_t *d = cube.data();   // all four ports read the cube; hls::burst_maxi takes a plain pointer
    justounetsimple_opt_readonly(d, d, d, d, dout.data(), h, w);

    uint16_t want[LANES];
    reference(cube, h, w, want);
    bool ok = true;
    for (int b = 0; b < LANES; b++) {
        const uint16_t got = (uint16_t)((uint32_t)dout[0][b / 2] >> (16 * (b % 2)));
        if (got != want[b]) {
            std::printf("  %d x %d: lane %d is %u, expected %u\n", h, w, b, got, want[b]);
            ok = false;
        }
    }
    for (int i = 0; i < 4; i++)
        if (dout[1][i] != SENTINEL) {
            std::printf("  %d x %d: wrote past the checksum\n", h, w);
            ok = false;
            break;
        }
    std::printf("%s %d x %d\n", ok ? "PASS" : "FAIL", h, w);
    return ok;
}

int main() {
    // 40 x 50 matches the m_axi depth pragmas (2 x 2 patches, the edge ones partial);
    // the others are smaller: one whole patch, one pixel, a strip one patch high.
    const int sizes[][2] = {{40, 50}, {32, 32}, {1, 1}, {7, 70}};
    bool ok = true;
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
        ok = check(sizes[i][0], sizes[i][1], 11 + i) && ok;
    std::printf(ok ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    return ok ? 0 : 1;
}
