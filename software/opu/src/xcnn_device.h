#pragma once

#include "cnn_config.h"
#include "kernel.h"
#include <cstdint>

struct LabelValues {
  uint8_t sea;
  uint8_t land;
  uint8_t cloud;
  bool operator==(const LabelValues &other) const {
    return sea == other.sea && land == other.land && cloud == other.cloud;
  }
};

constexpr LabelValues default_label_values = {0, 1, 2};

void translate_class_values(std::string compatible_string);

class CnnDevice {
public:
  CnnDevice(CaptureConfig &capture_config, AddressConfig &addresses,
            std::string uio_name, std::string compatible);
  ~CnnDevice();
  void start();
  bool isReady();
  bool isDone();
  bool isIdle();
  uint64_t cycles();
  LabelValues getLabelValues() { return label_values; };

  // Do not allow copy of this class
  CnnDevice(const CnnDevice &) = delete;
  CnnDevice &operator=(const CnnDevice &) = delete;

private:
  KernelInstance device{};
  LabelValues label_values;
};
