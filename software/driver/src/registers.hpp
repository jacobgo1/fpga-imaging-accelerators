#pragma once

#include <cstdint>
#include <string>
#include <vector>

// The kernel's AXI-Lite registers, read at run time from the x<kernel>_hw.h
// that Vitis HLS writes next to the IP (the file the notebook reads too). A
// rebuilt kernel whose registers moved, or a Zynq-7030 version with fewer read
// ports, then needs no new driver: only its own header.
struct Register {
  uint32_t offset = 0;
  int bits = 32;
};

struct RegisterMap {
  uint32_t ap_ctrl = 0;
  std::vector<Register> din;   // din0, din1, ...: every read port of the cube
  Register dout, height, width;

  static RegisterMap from_header(const std::string &path);
  static RegisterMap from_text(const std::string &text);
};
