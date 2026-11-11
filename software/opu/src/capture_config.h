#pragma once
// The program's arguments, and a capture's size from its capture_config.ini
// (cnn-1d-picobob's cnn_config.h, less what the 1D CNN alone needs).

#include <cstdint>
#include <filesystem>
#include <optional>
#include <spdlog/spdlog.h>

struct ProgramConfig {
  std::filesystem::path input_file;          // binned_cube.bip, or a cube .npy
  std::filesystem::path output_file;         // the labels, H x W uint8
  std::optional<std::filesystem::path> capture_config_file; // the .bip's size
  std::optional<std::filesystem::path> scores_file;         // the kernel's scores, .npy
  std::optional<std::filesystem::path> truth_file;          // true labels, .npy
  spdlog::level::level_enum log_level;
};

struct CaptureConfig {
  uint32_t aoi_y;
  uint32_t frame_count; // the cube's height
  uint32_t row_count;   // its width
  float exposure;
};

bool setProgramConfig(int argc, char *argv[], ProgramConfig &config);
CaptureConfig readCaptureConfig(std::filesystem::path filePath);
