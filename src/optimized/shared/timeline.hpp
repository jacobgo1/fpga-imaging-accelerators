#pragma once

#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "patch_grid.hpp"

// A timeline of a DATAFLOW kernel, recorded in the hardware for the notebook.
// Every FIFO between two processes gets two taps, pass-through processes that
// watch the words go by:
//
//   producer ─► o (2) ─► tap "gave" ─► s (the FIFO's depth) ─► tap "took" ─► i (2) ─► consumer
//                           │                                     │
//                           └──────────── ev ─► timeline ◄── ev ──┘
//
// The FIFOs next to the producer and the consumer are 2 deep, so "gave" sees
// each word within a cycle or two of the producer writing it, and "took"
// within a cycle or two of the consumer reading it. The FIFO in the middle
// keeps the depth the kernel gave it. For every patch each tap marks its
// first and its last word, and it counts the cycles it found nothing to take
// (the FIFO before it empty) and the cycles it could not pass a word on (the
// FIFO after it full). One counter (timeline) stamps every mark. A tap moves
// a word per cycle, so it adds a few cycles of latency, not throughput.

// The mark of a word: a bit of the word itself. HLS orders operations only by
// what they depend on; a mark that did not depend on the word went out before
// the word had arrived (on the board, every patch's first word was marked
// when the tap started that patch).
template<typename E, int N>
uint32_t mark_of(const vec_t<E, N>& w) {
    #pragma HLS INLINE
    return (uint32_t)w.v[0] & 1;
}
template<typename W>
uint32_t mark_of(const W& w) {   // hls::vector
    #pragma HLS INLINE
    return (uint32_t)w[0] & 1;
}

// tap: in -> out unchanged, PER_PATCH words a patch (what the producer writes
// per patch). On ev: a mark at the first and at the last word of every patch,
// then, after the last patch, the cycles in was empty and the cycles out was
// full, from the tap's start (with the kernel's) to its last word.
template<int PH, int PW, int PER_PATCH, typename T>
void tap(hls::stream<T>& in, hls::stream<T>& out, hls::stream<uint32_t>& ev, int height, int width) {
    static_assert(PER_PATCH >= 2, "a first and a last word, in different cycles");
    const int n = patch_count<PH, PW>(height, width);
    uint32_t empty = 0, full = 0;
    int p = 0, k = 0;
    move: while (p < n) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3000000
        if (in.empty()) {
#ifndef __SYNTHESIS__
            // C simulation runs the processes one after another: what is not
            // there now never comes. A wrong PER_PATCH would wait forever.
            std::fprintf(stderr, "tap: the stream ended early (patch %d, word %d of %d)\n", p, k, PER_PATCH);
            std::abort();
#endif
            empty++;
        } else if (out.full()) {
            full++;
        } else {
            const T v = in.read();
            out.write(v);
            if (k == 0 || k == PER_PATCH - 1) ev.write(mark_of(v));
            if (k == PER_PATCH - 1) {
                k = 0;
                p++;
            } else {
                k++;
            }
        }
    }
    ev.write(empty);
    ev.write(full);
}

// The buffer timeline writes, uint32 words:
//   [0]                 the total: cycles from the start until finished
//   [1 + i]             lane i: its marks (2 per patch)
//   [32 + i], [64 + i]  lane i: cycles its FIFO before was empty, after it full
//   [HEAD + i * 2 * PATCHES + 2k, + 1]   lane i, patch k: first and last word's cycle
// Only the first PATCHES patches' times are kept; the counts say if there were more.
template<int LANES, int PATCHES>
struct timeline_layout {
    static constexpr int MARKS = 1, EMPTY = 32, FULL = 64, HEAD = 128;
    static constexpr int WORDS = HEAD + LANES * 2 * PATCHES;
    static_assert(LANES < 32, "the counts fit in the head");
};

// timeline: counts cycles from the kernel's start and stamps every mark on
// ev[i] with the cycle it arrived (a cycle or two after the tap made it, the
// same for every lane). It stops when finished has come (the total) and every
// tap has sent all of it: 2 marks a patch, then its two counts. Then it writes
// the buffer above to out. Each lane has its own memory, PATCHES x 2 x 32 bits.
//
// Put it last in the DATAFLOW region: C simulation runs the processes one
// after another, so it finds everything already waiting, stamps the marks
// with cycles 0, 1, 2, ..., the counts are 0 and so is the total. Only the
// RTL keeps time; the number of marks is right in both.
template<int PH, int PW, int LANES, int PATCHES>
void timeline(hls::stream<uint32_t> ev[LANES], hls::stream<bool>& finished, uint32_t *out,
              uint32_t& cycles, int height, int width) {
    typedef timeline_layout<LANES, PATCHES> L;
    const uint32_t marks = 2 * (uint32_t)patch_count<PH, PW>(height, width);
    uint32_t times[LANES][2 * PATCHES];
    #pragma HLS ARRAY_PARTITION variable=times dim=1 type=complete
    uint32_t got[LANES], empty[LANES], full[LANES];
    #pragma HLS ARRAY_PARTITION variable=got type=complete
    #pragma HLS ARRAY_PARTITION variable=empty type=complete
    #pragma HLS ARRAY_PARTITION variable=full type=complete
    for (int i = 0; i < LANES; i++) {
        #pragma HLS UNROLL
        got[i] = empty[i] = full[i] = 0;
    }
    uint32_t t = 0, total = 0;
    bool ended = false, more = true;
    tick: while (more) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3000000
        bool all_in = true, any = false;
        for (int i = 0; i < LANES; i++) {
            #pragma HLS UNROLL
            uint32_t w;
            if (ev[i].read_nb(w)) {
                any = true;
                if (got[i] < marks) {
                    if (got[i] < 2 * PATCHES) times[i][got[i]] = t;
                } else if (got[i] == marks) {
                    empty[i] = w;
                } else {
                    full[i] = w;
                }
                got[i]++;
            }
            if (got[i] != marks + 2) all_in = false;
        }
        bool f;
        if (!ended && finished.read_nb(f)) {
            ended = true;
            total = t;
        }
        more = !(ended && all_in);
#ifndef __SYNTHESIS__
        // C simulation: after finished, nothing more is coming.
        if (more && ended && !any) {
            std::fprintf(stderr, "timeline: a tap did not send its %u marks and two counts\n", (unsigned)marks);
            std::abort();
        }
#endif
        t++;
    }
    cycles = total;

    out[0] = total;
    head_marks: for (int i = 0; i < LANES; i++) {
        #pragma HLS PIPELINE II=1
        out[L::MARKS + i] = marks;
    }
    head_empty: for (int i = 0; i < LANES; i++) {
        #pragma HLS PIPELINE II=1
        out[L::EMPTY + i] = empty[i];
    }
    head_full: for (int i = 0; i < LANES; i++) {
        #pragma HLS PIPELINE II=1
        out[L::FULL + i] = full[i];
    }
    lanes: for (int i = 0; i < LANES; i++) {
        #pragma HLS UNROLL
        // i is a constant in each copy of this loop: it picks the memory.
        const int n = marks < 2 * PATCHES ? (int)marks : 2 * PATCHES;
        put: for (int k = 0; k < n; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=2 max=1330
            out[L::HEAD + i * 2 * PATCHES + k] = times[i][k];
        }
    }
}
