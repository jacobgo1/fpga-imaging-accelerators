#ifndef FPGA_ACCEL_HLS_BURST_MAXI_SHIM_HPP
#define FPGA_ACCEL_HLS_BURST_MAXI_SHIM_HPP

// hls::burst_maxi<T>: an m_axi port whose bursts the code spells out itself --
// write_request (address and length), write (the data, one word per call),
// write_response (wait for the memory's acknowledgement); read_request, then
// read (one word per call) -- so it decides when to ask and when to wait.
// Vitis HLS ships the real one (hls_burst_maxi.h). For main.py's "native"
// stage (plain g++) this is a pointer with the same calls, which also checks
// that they are used the way the hardware needs: no more words than requested,
// a new write request only after the previous one's words, by the end one
// response per write request (else the kernel could report done before its
// last words reached memory), and every requested word read. Read requests
// may run ahead of the reading (that is what they are for): the words come
// back in request order.
#if defined(__VITIS_HLS__) || defined(__SYNTHESIS__)
#include <hls_burst_maxi.h>
#else
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace hls {

template <typename T>
class burst_maxi {
public:
    burst_maxi(T* base) : base_(base) {}                   // a testbench passes a plain pointer
    burst_maxi(const burst_maxi& other) : base_(other.base_) {}   // a copy has no bursts of its own yet
    ~burst_maxi() {
        check(responses_ == requests_, "a write burst was never acknowledged (write_response)");
        check(reads_ == 0, "a read burst's words were never all read");
    }

    void write_request(size_t offset, unsigned len) {
        check(left_ == 0, "write_request before the previous burst's words were written");
        next_ = base_ + offset;
        left_ = len;
        requests_++;
    }
    void write(const T& value) {
        check(left_ > 0, "write beyond the requested burst");
        *next_++ = value;
        left_--;
    }
    void read_request(size_t offset, unsigned len) {
        check(len > 0, "read_request of no words");
        check(reads_ < MAX_READS, "more read requests outstanding than the shim keeps (16)");
        Burst& b = queue_[(head_ + reads_) % MAX_READS];
        b.next = base_ + offset;
        b.left = len;
        reads_++;
    }
    T read() {
        check(reads_ > 0, "read without a read_request");
        Burst& b = queue_[head_];
        const T value = *b.next++;
        if (--b.left == 0) {
            head_ = (head_ + 1) % MAX_READS;
            reads_--;
        }
        return value;
    }
    void write_response() {
        check(responses_ < requests_, "write_response without a request");
        check(left_ == 0 || responses_ + 1 < requests_, "write_response before the burst's words were written");
        responses_++;
    }

private:
    static void check(bool ok, const char* what) {
        if (!ok) {
            std::fprintf(stderr, "hls::burst_maxi: %s\n", what);
            std::abort();
        }
    }

    struct Burst {
        T* next;
        unsigned left;
    };
    static const int MAX_READS = 16;

    T* base_;
    T* next_ = nullptr;
    unsigned left_ = 0;          // words of the current burst still to write
    long requests_ = 0, responses_ = 0;
    Burst queue_[MAX_READS];     // read requests not yet read, oldest at head_
    int head_ = 0, reads_ = 0;
};

} // namespace hls
#endif

#endif // FPGA_ACCEL_HLS_BURST_MAXI_SHIM_HPP
