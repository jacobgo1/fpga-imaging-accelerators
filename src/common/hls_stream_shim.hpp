#ifndef FPGA_ACCEL_HLS_STREAM_SHIM_HPP
#define FPGA_ACCEL_HLS_STREAM_SHIM_HPP

// Vitis HLS has the real hls::stream. The native build (plain g++) gets this queue
// with the same read/write/empty: behaviour only, no timing.
#if defined(__VITIS_HLS__) || defined(__SYNTHESIS__)
#include <hls_stream.h>
#else
#include <cassert>
#include <queue>

namespace hls {

template <typename T>
class stream {
public:
    stream() = default;
    explicit stream(const char* /*name*/) {}

    void write(const T& value) { values_.push(value); }
    T read() {
        assert(!values_.empty() && "hls::stream read from an empty stream");
        T value = values_.front();
        values_.pop();
        return value;
    }
    bool empty() const { return values_.empty(); }

private:
    std::queue<T> values_;
};

} // namespace hls
#endif

#endif
