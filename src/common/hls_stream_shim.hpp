#ifndef FPGA_ACCEL_HLS_STREAM_SHIM_HPP
#define FPGA_ACCEL_HLS_STREAM_SHIM_HPP

// A real Vitis HLS run (csim/csynth/cosim) ships hls_stream.h and defines
// __VITIS_HLS__ throughout the flow (and __SYNTHESIS__ during csynth) -- use
// the real, synthesizable hls::stream there. main.py's "native" stage
// compiles kernels with plain g++/clang and no Vitis install at all (that is
// the whole point of "native" -- see README), so it needs something with the
// same read()/write()/empty() surface plain g++ can compile. A queue is
// functionally equivalent for the single-producer/single-consumer FIFO
// pattern every kernel here uses, which is all this shim promises to be:
// it is not a resource/timing model, only a behavioral one for csim-less
// testing.
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

    void write(const T& value) { data_.push(value); }
    T read() {
        assert(!data_.empty() && "hls::stream read from an empty stream");
        T value = data_.front();
        data_.pop();
        return value;
    }
    bool empty() const { return data_.empty(); }

private:
    std::queue<T> data_;
};

} // namespace hls
#endif

#endif // FPGA_ACCEL_HLS_STREAM_SHIM_HPP
