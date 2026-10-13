#include "accelerator.hpp"

#include <chrono>
#include <stdexcept>
#include <thread>

// AP_CTRL bits (any HLS kernel with s_axilite port=return).
constexpr uint32_t AP_START = 1u << 0;
constexpr uint32_t AP_DONE = 1u << 1;   // cleared when read
constexpr uint32_t AP_IDLE = 1u << 2;

bool Accelerator::idle() { return regs_.read(map_.ap_ctrl) & AP_IDLE; }

void Accelerator::set(const Register &reg, uint64_t value) {
  if (reg.bits <= 32 && (value >> 32)) {
    throw std::runtime_error("address above 4 GB for a 32-bit register");
  }
  regs_.write(reg.offset, static_cast<uint32_t>(value));
  if (reg.bits > 32) {
    regs_.write(reg.offset + 4, static_cast<uint32_t>(value >> 32));
  }
}

void Accelerator::start(uint64_t cube, uint64_t scores, uint32_t height, uint32_t width) {
  if (!idle()) {
    throw std::runtime_error("the kernel is not idle: is the bitstream loaded, or is it still running?");
  }
  for (const auto &din : map_.din) {
    set(din, cube);
  }
  set(map_.dout, scores);
  set(map_.height, height);
  set(map_.width, width);
  regs_.write(map_.ap_ctrl, AP_START);
}

void Accelerator::wait(double timeout_s) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  while (!(regs_.read(map_.ap_ctrl) & AP_DONE)) {
    if (std::chrono::steady_clock::now() > deadline) {
      throw std::runtime_error("the kernel did not finish within " + std::to_string(timeout_s) + " s");
    }
    // A strip takes tens of ms: a short sleep costs nothing measurable and
    // leaves the CPU to the copy of the next strip.
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}
