#pragma once

// hls::vector<T, N>: N values that HLS moves as one wide word, so an m_axi port
// of them reads N values per bus beat (a plain struct is read field by field).
// Vitis HLS ships the real one; main.py's "native" stage (plain g++, no Vitis)
// gets this stand-in with the same element access.
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
