#include "sim.hpp"

#include "formats.hpp"

// Which kernel stands in for the FPGA: CMake's JOPT_SIM_KERNEL.
#ifdef JOPT_SIM_ZYNQ7030
#include "justounetsimple_opt_zynq7030.hpp"
#else
#include "justounetsimple_opt.hpp"
#endif

#include <stdexcept>
#include <string>
#include <vector>

uint32_t SimRegisters::read(uint32_t offset) {
  if (offset == map_.ap_ctrl) {
    const uint32_t ctrl = 0x4 | (done_ ? 0x2 : 0);   // always idle between starts
    done_ = false;                                   // done clears when read
    return ctrl;
  }
  return regs_[offset];
}

void SimRegisters::write(uint32_t offset, uint32_t value) {
  if (offset == map_.ap_ctrl) {
    if (value & 1) run();
    return;
  }
  regs_[offset] = value;
}

uint64_t SimRegisters::value(const Register &reg) {
  uint64_t v = regs_[reg.offset];
  if (reg.bits > 32) v |= uint64_t(regs_[reg.offset + 4]) << 32;
  return v;
}

const void *SimRegisters::pointer(uint64_t address, size_t bytes) {
  // The FPGA would read whatever is there; here it must be inside the region.
  if (address < memory_.phys() || address + bytes > memory_.phys() + memory_.size()) {
    throw std::runtime_error("sim: the kernel would access outside the reserved memory");
  }
  return const_cast<const uint8_t *>(memory_.data() + (address - memory_.phys()));
}

void SimRegisters::run() {
  const int height = static_cast<int>(value(map_.height));
  const int width = static_cast<int>(value(map_.width));
  const size_t pixels = size_t(height) * width;
  // The kernel's ports, in order; a header with fewer than the kernel has
  // repeats them, so the test can check that every port is given the cube.
  std::vector<const jopt_raw_t *> din;
  for (size_t i = 0; i < 4; i++) {
    const Register &reg = map_.din[i % map_.din.size()];
    din.push_back(static_cast<const jopt_raw_t *>(pointer(value(reg), pixels * IN_PIXEL)));
  }
  auto *dout = const_cast<jopt_out_t *>(
      static_cast<const jopt_out_t *>(pointer(value(map_.dout), pixels * OUT_PIXEL)));
#ifdef JOPT_SIM_ZYNQ7030
  justounetsimple_opt_zynq7030(din[0], din[1], dout, height, width);
#else
  justounetsimple_opt(din[0], din[1], din[2], din[3], dout, height, width);
#endif
  done_ = true;
}
