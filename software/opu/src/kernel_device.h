#pragma once
// The HLS kernel through its HLS-generated driver (cnn-1d-picobob's xcnn_device.h, for
// any kernel): kernel.h maps the kernel's own driver functions to the names used here
// (made by make_kernel_header.sh from the driver in the kernel's export.zip).

#include "classify.h"
#include "kernel.h"
#include <chrono>
#include <cstdint>
#include <string>

class KernelDevice {
public:
  KernelDevice(std::string uio_name, std::string compatible);
  ~KernelDevice();
  // Starts the kernel on HEIGHT x WIDTH pixels and waits until it is done: its own
  // count of its cycles, 0 if it keeps none.
  uint64_t run(uint64_t cube_address, uint64_t scores_address, uint32_t height,
               uint32_t width, std::chrono::seconds timeout);
  bool isReady();
  bool isIdle();
  LabelValues getLabelValues() { return label_values; };

  // Do not allow copy of this class
  KernelDevice(const KernelDevice &) = delete;
  KernelDevice &operator=(const KernelDevice &) = delete;

private:
  KernelInstance device{};
  LabelValues label_values;
};
