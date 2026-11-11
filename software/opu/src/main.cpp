#include "cnn_config.h"
#include "uio_memory_device.h"
#include "xcnn_device.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <thread>

using dn_t = u16;
using output_t = u8;
// The kernel writes 4 int32 a pixel: its 3 scores, then the class
using score_t = int32_t;
constexpr size_t SCORES_OUT = 4;

std::string accelerator_name = "hls_kernel";
std::string accelerator_compatible = "hls-kernel-classes";
std::string memory_name = "hls_kernel_memory";

void checkBuffer(size_t required_size, UioMap &memory) {
  if (required_size > memory.maxSize()) {
    throw std::runtime_error("Buffer size " + std::to_string(required_size) +
                             " exceeds reserved memory size " +
                             std::to_string(memory.maxSize()) + " for map " +
                             memory.name());
  }
}

void checkFile(size_t required_size, const std::filesystem::path &path) {
  auto file_size = std::filesystem::file_size(path);
  if (file_size < required_size) {
    throw std::runtime_error("Trying to read " + std::to_string(required_size) +
                             " bytes from " + path.string() +
                             ", but it only contains " +
                             std::to_string(file_size) + " bytes");
  }
}

void readBipFile(std::filesystem::path bip_path, CaptureConfig &config,
                 UioMap &bip_memory) {
  auto n_bytes =
      (config.row_count * config.frame_count * BANDS_RAW) * sizeof(dn_t);
  checkBuffer(n_bytes, bip_memory);
  checkFile(n_bytes, bip_path);

  spdlog::debug("Loading {} bytes of BIP data from {}", n_bytes,
                bip_path.string());
  std::ifstream bip_file(bip_path, std::ios::binary);
  if (!bip_file) {
    throw std::runtime_error("Failed to read BIP file: " + bip_path.string());
  }
  bip_memory.mapMemory(n_bytes);
  bip_file.read(reinterpret_cast<char *>(bip_memory.data()), n_bytes);
  auto bytes_read = bip_file.gcount();
  bip_memory.unmapMemory();
  bip_file.close();
  if (bytes_read != static_cast<std::streamsize>(n_bytes)) {
    bip_memory.unmapMemory();
    throw std::runtime_error("Failed to read expected amount of data from BIP "
                             "file at offset 0: expected " +
                             std::to_string(n_bytes) + " but read " +
                             std::to_string(bytes_read));
  }
}

void writeOutputFile(std::filesystem::path output_path, CaptureConfig &config,
                     UioMap &output_memory, LabelValues label_values) {
  auto n_pixels = config.row_count * config.frame_count;
  auto n_bytes = n_pixels * sizeof(output_t);
  auto n_score_bytes = n_pixels * SCORES_OUT * sizeof(score_t);
  checkBuffer(n_score_bytes, output_memory);

  spdlog::debug("Writing {} bytes of output data to {}", n_bytes,
                output_path.string());
  std::ofstream output_file(output_path, std::ios::binary);
  if (!output_file) {
    throw std::runtime_error(
        "Failed to write output file: " + output_path.string() + ": " +
        std::string(strerror(errno)));
  }
  output_memory.mapMemory(n_score_bytes);

  // Keep each pixel's class, the last of its scores
  auto scores = reinterpret_cast<const score_t *>(output_memory.data());
  std::vector<output_t> labels(n_pixels);
  for (size_t i = 0; i < n_pixels; i++) {
    labels[i] = static_cast<output_t>(scores[i * SCORES_OUT + SCORES_OUT - 1]);
  }
  if (!(label_values == default_label_values)) {
    // The values of our classes do not match those that we want. Translate
    // before saving t o file
    for (size_t i = 0; i < n_bytes; i++) {
      uint8_t labelval = labels[i];
      if (labelval == label_values.sea) {
        labels[i] = default_label_values.sea;
      } else if (labelval == label_values.land) {
        labels[i] = default_label_values.land;
      } else {
        labels[i] = default_label_values.cloud;
      }
    }
  }
  output_file.write(reinterpret_cast<const char *>(labels.data()), n_bytes);
  output_memory.unmapMemory();
  if (!output_file) {
    throw std::runtime_error(
        "Failed to write output file: " + output_path.string() + ": " +
        std::string(strerror(errno)));
  }
  output_file.close();
}

void cnn_main(ProgramConfig &program_config) {

  std::chrono::steady_clock::time_point start_time;
  CaptureConfig capture_config;
  AddressConfig addresses;

  start_time = std::chrono::steady_clock::now();
  capture_config = readCaptureConfig(program_config.capture_config_file);
  spdlog::debug("Configuration loaded: rows={}, frames={}, aoi_y={}, "
                "exposure={}",
                capture_config.row_count, capture_config.frame_count,
                capture_config.aoi_y, capture_config.exposure);

  UioDevice cnn_1d_memory(memory_name);
  if (cnn_1d_memory.getNumMaps() < 2) {
    throw std::runtime_error(
        "Expected at least 2 maps for " + memory_name + " but found " +
        std::to_string(cnn_1d_memory.getNumMaps()));
  }
  UioMap dn_memory = cnn_1d_memory.getMap(0);
  UioMap output_memory = cnn_1d_memory.getMap(1);

  start_time = std::chrono::steady_clock::now();
  addresses.dn_address = dn_memory.physicalAddress();
  readBipFile(program_config.bip_file, capture_config, dn_memory);
  auto data_load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start_time)
                            .count();
  spdlog::info("Input data loaded in {} ms", data_load_time);

  start_time = std::chrono::steady_clock::now();
  addresses.output_address = output_memory.physicalAddress();
  CnnDevice cnn(capture_config, addresses, accelerator_name,
                accelerator_compatible);

  if (!cnn.isReady()) {
    throw std::runtime_error("CNN device not ready to start");
  }

  const auto max_runtime = std::chrono::seconds(30);
  auto cnn_start_time = std::chrono::steady_clock::now();
  cnn.start();
  do {
    if (std::chrono::steady_clock::now() - start_time > max_runtime) {
      throw std::runtime_error("Timeout waiting for device to finish");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } while (!cnn.isIdle());
  auto cnn_runtime = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - cnn_start_time)
                         .count();
  spdlog::info("CNN finished in {} ms", cnn_runtime);
  if (kernelHasCycles()) {
    spdlog::info("CNN's own count: {} cycles", cnn.cycles());
  }
  writeOutputFile(program_config.output_file, capture_config, output_memory,
                  cnn.getLabelValues());

  auto runtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time)
                        .count();
  spdlog::info("CNN processing completed and output written in {} ms",
               runtime_ms);
}

int main(int argc, char *argv[]) {
  auto program_start = std::chrono::steady_clock::now();

  // Read command line arguments and store them in a config struct
  ProgramConfig program_config;
  bool program_config_valid = setProgramConfig(argc, argv, program_config);
  if (!program_config_valid) {
    return 1;
  }

  try {
    cnn_main(program_config);
  } catch (const std::exception &e) {
    spdlog::error("Error: {}", e.what());
    return 1;
  }

  const auto total_duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - program_start);
  spdlog::info("Total runtime {} ms", total_duration_ms.count());
  return 0;
}
