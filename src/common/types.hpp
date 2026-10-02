#ifndef COMMON_TYPES_HPP
#define COMMON_TYPES_HPP

#include <cstdint>
#include <cstddef>

#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__) || defined(AESL_TB)
#include <hls_stream.h>
#else
// Lightweight STL-based fallback for native C++ testbenches (g++ / clang++)
#if __has_include(<hls_stream.h>)
#include <hls_stream.h>
#else
#include <queue>
#include <string>
#include <stdexcept>

namespace hls {
template <typename T>
class stream {
public:
    stream() = default;
    stream(const char* /*name*/) {}
    stream(const std::string& /*name*/) {}

    bool empty() const { return q_.empty(); }
    bool full() const { return false; }
    size_t size() const { return q_.size(); }

    void write(const T& val) { q_.push(val); }
    bool write_nb(const T& val) { q_.push(val); return true; }

    T read() {
        if (q_.empty()) {
            return T();
        }
        T val = q_.front();
        q_.pop();
        return val;
    }

    bool read_nb(T& val) {
        if (q_.empty()) return false;
        val = q_.front();
        q_.pop();
        return true;
    }

    void operator>>(T& val) { val = read(); }
    void operator<<(const T& val) { write(val); }

private:
    std::queue<T> q_;
};
} // namespace hls
#endif
#endif

// Standard integer and fixed-point precisions for imaging accelerators
using data_t   = std::int16_t;   // Input pixels and weights (16-bit)
using acc_t    = std::int32_t;   // Accumulator (prevents 64-bit adder-tree explosion)
using result_t = std::int16_t;   // Quantized/scaled output activations

#endif // COMMON_TYPES_HPP
