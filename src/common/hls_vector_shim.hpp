#pragma once

// N values HLS moves as one wide word, so an m_axi port reads N per bus beat.
// Vitis HLS has the real one; the native build gets this stand-in.
#if defined(__VITIS_HLS__) || defined(__SYNTHESIS__)
#include <hls_vector.h>
#else
namespace hls {
template<typename T, unsigned N>
struct vector {
    T data[N];
    T& operator[](unsigned i) { return data[i]; }
    const T& operator[](unsigned i) const { return data[i]; }
};
}  // namespace hls
#endif
