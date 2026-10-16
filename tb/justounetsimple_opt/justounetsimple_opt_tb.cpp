// justounetsimple_opt_tb.cpp  --  testbench for the optimized kernel. The
// checks are in justounetsimple_opt_check.hpp, shared with the Zynq-7030 one;
// the timeline (trace) is this kernel's own and checked here.
#include "justounetsimple_opt.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

static void run_kernel(jopt_raw_t *din, jopt_out_t *dout, int height, int width) {
    // All four ports read the cube; hls::burst_maxi takes a plain (not const) pointer.
    // cycles and the trace's times count only in the RTL (cosim): in C
    // simulation cycles is 0 and the times are just the marks' order.
    uint32_t cycles = 0;
    std::vector<uint32_t> trace(TRACE_WORDS + 1, 0xdeadbeef);   // one more: nothing written past the end
    justounetsimple_opt(din, din, din, din, dout, height, width, cycles, trace.data());
    std::printf("cycles register: %u, trace total: %u\n", (unsigned)cycles, (unsigned)trace[0]);

    // Every tap passed exactly its words and sent two marks per patch and its
    // two counts on every lane: a wrong count stops the tap or the timeline
    // with an error before this. The empty and full counts are written: 0 in
    // C simulation (cycles 0, nothing waits), less than the run in cosim.
    const int patches = ((height + H - 1) / H) * ((width + W - 1) / W);
    bool ok = trace[0] == cycles && trace[TRACE_WORDS] == 0xdeadbeef;
    for (int i = 0; i < TRACE_LANES; i++) {
        const uint32_t marks = trace[1 + i], empty = trace[32 + i], full = trace[64 + i];
        const bool waits_ok = cycles == 0 ? empty == 0 && full == 0 : (uint64_t)empty + full < cycles;
        if (marks != 2u * patches || !waits_ok) {
            std::printf("trace lane %d: %u marks (expected %d), empty %u, full %u, of %u cycles\n", i,
                        (unsigned)marks, 2 * patches, (unsigned)empty, (unsigned)full, (unsigned)cycles);
            ok = false;
        }
        for (int k = 1; k < (int)marks && k < 2 * TRACE_PATCHES; k++)
            if (trace[TRACE_HEAD + i * 2 * TRACE_PATCHES + k] < trace[TRACE_HEAD + i * 2 * TRACE_PATCHES + k - 1]) {
                std::printf("trace lane %d: mark %d before mark %d\n", i, k, k - 1);
                ok = false;
            }
    }
    std::printf("trace: %s (%d lanes, %d patches)\n", ok ? "PASS" : "FAIL", TRACE_LANES, patches);
    if (!ok) std::exit(1);
}

#include "justounetsimple_opt_check.hpp"
