// justoliunet_tb.cpp  --  testbench for the justoliunet kernel, not a
// synthesis source. The reference is the same justoliunet_golden<float,...>
// call the kernel wraps (see justoliunet.hpp part 1): this checks that
// wiring concrete geometry through interface pragmas didn't change what
// gets computed, and gives a later variant (e.g. once data_t becomes a
// fixed-point type) a place to grow a real, numerically-independent
// tolerance check instead of exact equality.
#include "justoliunet.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

static data_t din[jnet::H][jnet::W][1];

static data_t w1[jnet::base_ch][1][jnet::K];
static data_t b1[jnet::base_ch];
static data_t w2[2 * jnet::base_ch][jnet::base_ch][jnet::K];
static data_t b2[2 * jnet::base_ch];
static data_t w3[3 * jnet::base_ch][2 * jnet::base_ch][jnet::K];
static data_t b3[3 * jnet::base_ch];
static data_t w4[4 * jnet::base_ch][3 * jnet::base_ch][jnet::K];
static data_t b4[4 * jnet::base_ch];
static data_t w5[jnet::num_classes][jnet::flat];
static data_t b5[jnet::num_classes];

static data_t dout[jnet::H][jnet::num_classes];
static data_t expected[jnet::H][jnet::num_classes];

// Arrays declared above have no padding between elements, so filling them
// as a flat run of `count` values is exactly filling them element by
// element in row-major order -- just without writing out five nested loops
// for five differently-shaped weight tensors.
static void fill_uniform(data_t *data, int count, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < count; i++)
        data[i] = dist(rng);
}

int main()
{
    std::mt19937 rng(1);                              // fixed seed: repeatable

    fill_uniform(&din[0][0][0], jnet::H * jnet::W, rng);
    fill_uniform(&w1[0][0][0], jnet::base_ch * 1 * jnet::K, rng);
    fill_uniform(b1, jnet::base_ch, rng);
    fill_uniform(&w2[0][0][0], 2 * jnet::base_ch * jnet::base_ch * jnet::K, rng);
    fill_uniform(b2, 2 * jnet::base_ch, rng);
    fill_uniform(&w3[0][0][0], 3 * jnet::base_ch * 2 * jnet::base_ch * jnet::K, rng);
    fill_uniform(b3, 3 * jnet::base_ch, rng);
    fill_uniform(&w4[0][0][0], 4 * jnet::base_ch * 3 * jnet::base_ch * jnet::K, rng);
    fill_uniform(b4, 4 * jnet::base_ch, rng);
    fill_uniform(&w5[0][0], jnet::num_classes * jnet::flat, rng);
    fill_uniform(b5, jnet::num_classes, rng);

    justoliunet_golden<data_t, jnet::H, jnet::W, jnet::K, jnet::base_ch, jnet::num_classes>(
        din, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, expected);

    justoliunet(din, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, dout);

    int errors = 0;
    for (int row = 0; row < jnet::H; row++)
        for (int cls = 0; cls < jnet::num_classes; cls++)
            if (dout[row][cls] != expected[row][cls]) {
                if (errors < 5)
                    printf("mismatch at [%d][%d]: got %g, expected %g\n",
                           row, cls, (double)dout[row][cls], (double)expected[row][cls]);
                errors++;
            }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", jnet::H * jnet::num_classes);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    // Timing, not correctness: only meaningful once PASS has already proven
    // dout is right. Same din/weights every call (steady-state throughput on
    // a fixed workload, not first-call/cache-cold latency) -- fine for
    // comparing this A53 number against an HLS csim/cosim/hardware number
    // for the *same* justoliunet(), not for modeling a real capture's mix of
    // inputs. Rep count is a knob, not a recompile: JNET_BENCH_REPS=0 skips
    // it, a higher count tightens the estimate on a noisy board.
    if (errors == 0) {
        int reps = 100;
        if (const char *env = std::getenv("JNET_BENCH_REPS"))
            reps = std::atoi(env);
        if (reps > 0) {
            double best_us = -1.0, total_us = 0.0;
            for (int rep = 0; rep < reps; rep++) {
                auto start = std::chrono::steady_clock::now();
                justoliunet(din, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, dout);
                auto end = std::chrono::steady_clock::now();
                double us = std::chrono::duration<double, std::micro>(end - start).count();
                total_us += us;
                if (best_us < 0.0 || us < best_us)
                    best_us = us;
            }
            double mean_us = total_us / reps;
            printf("BENCH  %d reps, %d rows/call: best %.1f us, mean %.1f us (%.0f rows/s)\n",
                   reps, jnet::H, best_us, mean_us, mean_us > 0.0 ? 1e6 * jnet::H / mean_us : 0.0);
        }
    }

    return errors == 0 ? 0 : 1;
}
