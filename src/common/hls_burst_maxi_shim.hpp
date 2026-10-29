#ifndef FPGA_ACCEL_HLS_BURST_MAXI_SHIM_HPP
#define FPGA_ACCEL_HLS_BURST_MAXI_SHIM_HPP

// An m_axi port whose bursts the kernel requests itself. Vitis HLS has the real
// one; the native build gets this pointer with the same calls, which also checks
// they are used as the hardware needs: no word beyond a request, every response
// waited for (or the kernel could finish before its last words reach memory),
// every requested word read. Read requests may run ahead of the reads.
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
    burst_maxi(T* base) : base_(base) {}
    burst_maxi(const burst_maxi& other) : base_(other.base_) {}   // a copy starts with no bursts
    ~burst_maxi() {
        check(write_responses_ == write_requests_, "a write burst was never acknowledged (write_response)");
        check(pending_reads_ == 0, "a read burst's words were never all read");
    }

    void write_request(size_t offset, unsigned length) {
        check(words_to_write_ == 0, "write_request before the previous burst's words were written");
        next_write_ = base_ + offset;
        words_to_write_ = length;
        write_requests_++;
    }
    void write(const T& value) {
        check(words_to_write_ > 0, "write beyond the requested burst");
        *next_write_++ = value;
        words_to_write_--;
    }
    void write_response() {
        check(write_responses_ < write_requests_, "write_response without a request");
        check(words_to_write_ == 0 || write_responses_ + 1 < write_requests_,
              "write_response before the burst's words were written");
        write_responses_++;
    }

    void read_request(size_t offset, unsigned length) {
        check(length > 0, "read_request of no words");
        check(pending_reads_ < MAX_PENDING_READS, "more read requests outstanding than the shim keeps (16)");
        ReadBurst& burst = read_queue_[(oldest_read_ + pending_reads_) % MAX_PENDING_READS];
        burst.next = base_ + offset;
        burst.words_left = length;
        pending_reads_++;
    }
    T read() {
        check(pending_reads_ > 0, "read without a read_request");
        ReadBurst& burst = read_queue_[oldest_read_];
        const T value = *burst.next++;
        if (--burst.words_left == 0) {
            oldest_read_ = (oldest_read_ + 1) % MAX_PENDING_READS;
            pending_reads_--;
        }
        return value;
    }

private:
    static void check(bool ok, const char* what) {
        if (!ok) {
            std::fprintf(stderr, "hls::burst_maxi: %s\n", what);
            std::abort();
        }
    }

    struct ReadBurst {
        T* next;
        unsigned words_left;
    };
    static const int MAX_PENDING_READS = 16;

    T* base_;
    T* next_write_ = nullptr;
    unsigned words_to_write_ = 0;
    long write_requests_ = 0, write_responses_ = 0;
    ReadBurst read_queue_[MAX_PENDING_READS];
    int oldest_read_ = 0, pending_reads_ = 0;
};

} // namespace hls
#endif

#endif
