// classify: runs an HLS kernel on a capture on the OPU system's PetaLinux (PicoBoB,
// the lab's ZedBoards), as cnn-1d-picobob's cnn_accelerator runs the 1D CNN. Every
// optimized kernel takes the raw cube and gives the same scores, so one program serves
// all; scripts/opu.sh builds it with the kernel's own HLS driver.
//
//   classify --input binned_cube.bip --capture-config capture_config.ini --output labels.bin
//   classify --input cube.npy --output labels.bin [--labels labels.npy] [--scores scores.npy]
//
// The kernel's registers come from UIO (hls_kernel), its buffers from the reserved
// memory (hls_kernel_memory: the cube map and the scores map, both in the overlay).
// UIO maps them uncached, so the kernel reads what the ARM wrote and the ARM what the
// kernel wrote, without cache flushes.
#include "capture_config.h"
#include "classify.h"
#include "kernel_device.h"
#include "uio_memory_device.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <spdlog/spdlog.h>

std::string kernel_uio_name = "hls_kernel";
std::string kernel_compatible = "hls-kernel-classes";
std::string memory_name = "hls_kernel_memory";
const auto max_runtime = std::chrono::seconds(60);

void kernel_main(ProgramConfig &program_config) {
  long height = 0, width = 0;
  if (program_config.capture_config_file) {
    auto capture_config = readCaptureConfig(*program_config.capture_config_file);
    spdlog::debug("Capture config loaded: rows={}, frames={}, aoi_y={}, "
                  "exposure={}",
                  capture_config.row_count, capture_config.frame_count,
                  capture_config.aoi_y, capture_config.exposure);
    height = capture_config.frame_count;
    width = capture_config.row_count;
  }
  Cube cube = openCube(program_config.input_file, height, width);
  spdlog::info("Cube {}: {} x {} pixels", program_config.input_file.string(),
               cube.height, cube.width);

  UioDevice memory(memory_name);
  if (memory.getNumMaps() < 2) {
    throw std::runtime_error("Expected at least 2 maps for " + memory_name +
                             " but found " +
                             std::to_string(memory.getNumMaps()));
  }
  UioMap cube_memory = memory.getMap(0);
  UioMap scores_memory = memory.getMap(1);
  cube_memory.mapMemory(cube_memory.maxSize());
  scores_memory.mapMemory(scores_memory.maxSize());

  KernelDevice kernel(kernel_uio_name, kernel_compatible);
  if (!kernel.isIdle()) {
    throw std::runtime_error("Kernel not idle: is another program using it?");
  }

  Outputs outputs;
  outputs.labels = program_config.output_file;
  outputs.scores = program_config.scores_file;
  outputs.truth = program_config.truth_file;
  outputs.kernel_values = kernel.getLabelValues();
  auto start = [&](uint64_t cube_address, uint64_t scores_address,
                   uint32_t h, uint32_t w) {
    return kernel.run(cube_address, scores_address, h, w, max_runtime);
  };
  Result result = classify(
      cube,
      {reinterpret_cast<uint8_t *>(cube_memory.data()),
       cube_memory.physicalAddress(), cube_memory.size()},
      {reinterpret_cast<uint8_t *>(scores_memory.data()),
       scores_memory.physicalAddress(), scores_memory.size()},
      start, outputs);
  result.has_cycles = kernelHasCycles();
  spdlog::info("Input data loaded in {:.0f} ms", 1e3 * result.load_s);
  spdlog::info("Kernel finished in {:.1f} ms, {} start(s)", 1e3 * result.fpga_s,
               result.starts);
  spdlog::info("Labels written to {} in {:.0f} ms",
               program_config.output_file.string(), 1e3 * result.save_s);
  std::printf("kernel:             %s, built for %.3f MHz\n", KERNEL_NAME,
              static_cast<double>(KERNEL_CLOCK_MHZ));
  printResult(result, outputs, std::cout);
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
    kernel_main(program_config);
  } catch (const std::exception &e) {
    spdlog::error("Error: {}", e.what());
    std::fprintf(stderr, "classify: %s\n", e.what());
    return 1;
  }

  const auto total = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - program_start)
                         .count();
  spdlog::info("Total runtime {:.0f} ms", 1e3 * total);
  std::printf("wall clock:         %8.3f s\n", total);
  return 0;
}
