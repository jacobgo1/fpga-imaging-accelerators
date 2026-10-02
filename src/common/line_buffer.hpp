#ifndef COMMON_LINE_BUFFER_HPP
#define COMMON_LINE_BUFFER_HPP

#include "types.hpp"

// ============================================================================
// LineBuffer
// Buffers (K - 1) rows of width WIDTH for streaming 2D convolutions/pooling.
// Fully partitioned along the line dimension so all lines can be read/written
// simultaneously in a single clock cycle.
// ============================================================================
template <typename T, int LINES, int WIDTH>
class LineBuffer {
public:
    T buffer[LINES][WIDTH];

    LineBuffer() {
        #pragma HLS ARRAY_PARTITION variable=buffer complete dim=1
    }

    T get(int line, int col) const {
        #pragma HLS INLINE
        return buffer[line][col];
    }

    void shift_up(int col, T new_val) {
        #pragma HLS INLINE
        for (int l = 0; l < LINES - 1; ++l) {
            #pragma HLS UNROLL
            buffer[l][col] = buffer[l + 1][col];
        }
        buffer[LINES - 1][col] = new_val;
    }
};

// ============================================================================
// WindowBuffer
// 2D register window of size HEIGHT x WIDTH.
// Fully partitioned into discrete flip-flops/registers (dim=0) allowing all
// K x K window values to be read simultaneously by unrolled MAC units.
// ============================================================================
template <typename T, int HEIGHT, int WIDTH>
class WindowBuffer {
public:
    T window[HEIGHT][WIDTH];

    WindowBuffer() {
        #pragma HLS ARRAY_PARTITION variable=window complete dim=0
    }

    T get(int row, int col) const {
        #pragma HLS INLINE
        return window[row][col];
    }

    void shift_left_insert_col(const T new_col[HEIGHT]) {
        #pragma HLS INLINE
        for (int r = 0; r < HEIGHT; ++r) {
            #pragma HLS UNROLL
            for (int c = 0; c < WIDTH - 1; ++c) {
                #pragma HLS UNROLL
                window[r][c] = window[r][c + 1];
            }
            window[r][WIDTH - 1] = new_col[r];
        }
    }
};

#endif // COMMON_LINE_BUFFER_HPP
