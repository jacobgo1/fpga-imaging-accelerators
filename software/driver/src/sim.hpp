#pragma once

#include "device.hpp"
#include "registers.hpp"

#include <map>

// The --sim build: the kernel's own C++ (src/hls/justounetsimple_opt/) stands
// in for the FPGA, on a PC. Writing the start bit runs it on the "DDR" region,
// with the addresses and sizes the driver wrote, then raises done and idle. It
// tests everything the driver does on the board except the hardware: the
// register traffic, the strips, the copies and the files.
class SimRegisters : public Registers {
public:
  SimRegisters(Region &memory, RegisterMap map) : memory_(memory), map_(std::move(map)) {}
  uint32_t read(uint32_t offset) override;
  void write(uint32_t offset, uint32_t value) override;

private:
  uint64_t value(const Register &reg);
  const void *pointer(uint64_t address, size_t bytes);
  void run();
  Region &memory_;
  RegisterMap map_;
  std::map<uint32_t, uint32_t> regs_;
  bool done_ = false;
};
