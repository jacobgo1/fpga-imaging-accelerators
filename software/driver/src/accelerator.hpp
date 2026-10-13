#pragma once

#include "device.hpp"
#include "registers.hpp"

#include <cstdint>

// justounetsimple_opt (or its Zynq-7030 version) through its control
// registers: every din port reads the same raw cube, [height][width][120] uint16, and
// dout gets [height][width][4] int32: 3 scores * 2^16 and the class.
class Accelerator {
public:
  Accelerator(Registers &registers, RegisterMap map) : regs_(registers), map_(std::move(map)) {}

  bool idle();
  void start(uint64_t cube, uint64_t scores, uint32_t height, uint32_t width);
  // Polls the done bit; throws after timeout_s seconds.
  void wait(double timeout_s);

  size_t read_ports() const { return map_.din.size(); }

private:
  void set(const Register &reg, uint64_t value);
  Registers &regs_;
  RegisterMap map_;
};
