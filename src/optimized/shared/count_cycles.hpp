#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"

// count_cycles: the kernel's own stopwatch, for the notebook. A DATAFLOW
// process that waits for no input, so it starts with the kernel, and counts
// one per clock cycle until the last process (store_scores) says its last
// word is in memory. cycles is then the kernel's time in cycles, a few cycles
// over at most, without the ARM's register writes and polling in it. It runs
// beside the others and takes nothing from them: a counter and a FIFO.
// 32 bits: 21 s at 200 MHz.
//
// Put it last in the DATAFLOW region. C simulation (and main.py native) runs
// the processes one after another, in order, so it gets there with finished
// already written and counts 0: only the RTL (cosim, the board) counts.
inline void count_cycles(hls::stream<bool>& finished, uint32_t& cycles) {
    #pragma HLS INLINE off
    uint32_t n = 0;
    tick: while (finished.empty()) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3000000
        n++;
    }
    finished.read();
    cycles = n;
}
