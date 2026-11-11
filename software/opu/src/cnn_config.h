#pragma once

#include <filesystem>
#include <spdlog/spdlog.h>

// Fixed instrument constants
#define BANDS_RAW 120
#define RAD_FULL_ROWS 1216
#define CLASSES_OUT 3

struct ProgramConfig {
  std::filesystem::path bip_file;
  std::filesystem::path output_file;
  std::filesystem::path capture_config_file;
  spdlog::level::level_enum log_level;
};

struct AddressConfig {
  uint64_t dn_address;
  uint64_t output_address;
};

struct CaptureConfig {
  uint32_t aoi_y;
  uint32_t frame_count;
  uint32_t row_count;
  float exposure;
};

bool setProgramConfig(int argc, char *argv[], ProgramConfig &config);
CaptureConfig readCaptureConfig(std::filesystem::path filePath);
