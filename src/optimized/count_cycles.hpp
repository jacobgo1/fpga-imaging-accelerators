#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"

// The kernel's stopwatch: a process that starts with the kernel and counts clock
// cycles until the last process says its last word is in memory.
// Put it last: C simulation runs the processes in order, so there it counts 0.
inline void count_cycles(hls::stream<bool>& finished, uint32_t& cycles) {
    #pragma HLS INLINE off
    uint32_t count = 0;
    tick: while (finished.empty()) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3000000
        count++;
    }
    finished.read();
    cycles = count;
}
