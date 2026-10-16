#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "patch_grid.hpp"

// A timeline of a DATAFLOW kernel, recorded in the hardware for the notebook:
// for every FIFO between two processes and every patch, the clock cycle its
// first word went through and the cycle its last did.
//
//   producer ─► tap ─► consumer       each tap marks first / last word of a patch
//                │
//                └─ ev ─► timeline    one cycle counter stamps every mark;
//                                     at the end, all of it to DDR
//
// tap is a pass-through process put right behind a producer, so it sees the
// producer's words as they come out: one word per cycle in and out, whatever
// the type, and the FIFO behind it keeps the depth the kernel gave it. It
// costs the word's registers and adds a cycle or two of latency, not
// throughput.

// tap: in -> out unchanged; on ev, true at the first and false at the last of
// every PER_PATCH words. PER_PATCH is what the producer writes per patch.
template<int PH, int PW, int PER_PATCH, typename T>
void tap(hls::stream<T>& in, hls::stream<T>& out, hls::stream<bool>& ev, int height, int width) {
    static_assert(PER_PATCH >= 2, "a first and a last word, in different cycles");
    const int n = patch_count<PH, PW>(height, width);
    patch: for (int p = 0; p < n; p++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        word: for (int k = 0; k < PER_PATCH; k++) {
            #pragma HLS PIPELINE II=1
            out.write(in.read());
            if (k == 0) ev.write(true);
            if (k == PER_PATCH - 1) ev.write(false);
        }
    }
}

// The buffer timeline writes, uint32 words:
//   [0]              the total: cycles from the start until finished (as count_cycles)
//   [1 .. LANES]     how many marks each lane made (2 per patch)
//   [HEAD + i * 2 * PATCHES + 2k, + 1]   lane i, patch k: first and last word's cycle
// Only the first PATCHES patches are kept; the counts say if there were more.
template<int LANES, int PATCHES>
struct timeline_layout {
    static constexpr int HEAD = 16;
    static constexpr int WORDS = HEAD + LANES * 2 * PATCHES;
    static_assert(1 + LANES <= HEAD, "the counts fit in the head");
};

// timeline: counts cycles from the kernel's start, like count_cycles, and
// stamps every mark on ev[i] with the cycle it arrived (a cycle or two after
// the tap made it, the same for every lane). It stops when finished has
// come and every mark is in; then it writes the buffer above to out. Each
// lane has its own memory (one write per cycle each), PATCHES x 2 x 32 bits.
//
// Put it last in the DATAFLOW region: C simulation runs the processes one
// after another, so it finds every mark already waiting, stamps them all
// with cycles 0, 1, 2, ... and the total is 0. Only the RTL keeps time; the
// counts are right in both.
template<int LANES, int PATCHES>
void timeline(hls::stream<bool> ev[LANES], hls::stream<bool>& finished, uint32_t *out, uint32_t& cycles) {
    typedef timeline_layout<LANES, PATCHES> L;
    uint32_t times[LANES][2 * PATCHES];
    #pragma HLS ARRAY_PARTITION variable=times dim=1 type=complete
    uint32_t marks[LANES];
    #pragma HLS ARRAY_PARTITION variable=marks type=complete
    for (int i = 0; i < LANES; i++) {
        #pragma HLS UNROLL
        marks[i] = 0;
    }
    uint32_t t = 0, total = 0;
    bool ended = false, more = true;
    tick: while (more) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3000000
        bool any = false;
        for (int i = 0; i < LANES; i++) {
            #pragma HLS UNROLL
            bool first;
            if (ev[i].read_nb(first)) {
                if (marks[i] < 2 * PATCHES) times[i][marks[i]] = t;
                marks[i]++;
                any = true;
            }
        }
        bool f;
        if (!ended && finished.read_nb(f)) {
            ended = true;
            total = t;
        }
        // Every tap marked its last word before store_scores could finish, so
        // after finished only marks still in their FIFOs remain.
        more = !ended || any;
        t++;
    }
    cycles = total;

    out[0] = total;
    head: for (int i = 0; i < LANES; i++) {
        #pragma HLS PIPELINE II=1
        out[1 + i] = marks[i];
    }
    lanes: for (int i = 0; i < LANES; i++) {
        #pragma HLS UNROLL
        // i is a constant in each copy of this loop: it picks the memory.
        const int n = marks[i] < 2 * PATCHES ? (int)marks[i] : 2 * PATCHES;
        put: for (int k = 0; k < n; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=2 max=1330
            out[L::HEAD + i * 2 * PATCHES + k] = times[i][k];
        }
    }
}
