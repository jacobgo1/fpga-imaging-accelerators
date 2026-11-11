#include "kernel_device.h"
#include <byteswap.h>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <stdexcept>

/**
 * Recursively look through device-tree for file called "compatible" and return
 * if it is compatible with search_string
 */
std::filesystem::path find_compatible_node(std::string search_string) {
  std::filesystem::path device_tree_path = "/proc/device-tree/";
  for (const auto &dir_entry :
       std::filesystem::recursive_directory_iterator(device_tree_path)) {
    if (!dir_entry.is_regular_file()) {
      continue;
    }
    auto filename = dir_entry.path().filename();
    if (filename.string() != std::string("compatible")) {
      continue;
    }

    auto compatible_file_path = dir_entry.path();
    std::ifstream compatible_file(compatible_file_path);

    if (!compatible_file.is_open()) {
      spdlog::warn("Could not open file {}", compatible_file_path.string());
      continue;
    }

    std::string file_content;
    std::getline(compatible_file, file_content);
    compatible_file.close();
    if (file_content.find(search_string) != std::string::npos) {
      auto node = compatible_file_path.parent_path();
      return node;
    }
  }
  return {};
}

/**
 * Get value of label class at device-tree node
 */
int get_class_value(const std::filesystem::path &node_path,
                    std::string class_name) {
  std::string class_file_name = "class-" + class_name;
  auto class_file_path = node_path / class_file_name;
  if (!std::filesystem::exists(class_file_path)) {
    return -1;
  }
  std::ifstream class_file(class_file_path, std::ios::binary);
  if (!class_file) {
    spdlog::warn("Could not open class file " + class_file_path.string());
    return -1;
  }

  uint32_t result;
  class_file.read(reinterpret_cast<char *>(&result), sizeof(result));
  return bswap_32(result);
}

const char *kernel_strerror(int errcode) {
  switch (errcode) {
  case XST_SUCCESS:
    return "Success";
  case XIL_COMPONENT_IS_READY:
    return "Component is ready";
  case XST_DEVICE_NOT_FOUND:
    return "Device not found";
  case XST_OPEN_DEVICE_FAILED:
    return "Failed to open device";
  default:
    return "Unknown error code";
  }
}

KernelDevice::KernelDevice(std::string uio_name, std::string compatible) {
  int ret = kernelInitialize(&device, uio_name.c_str());
  if (ret != XST_SUCCESS) {
    throw std::runtime_error("Initializing " + std::string(KERNEL_NAME) +
                             " (UIO " + uio_name + ") failed: " +
                             std::string(kernel_strerror(ret)) +
                             ". Is its bitstream and overlay loaded?");
  }

  auto node = find_compatible_node(compatible);
  auto sea_class = node.empty() ? -1 : get_class_value(node, "sea");
  auto land_class = node.empty() ? -1 : get_class_value(node, "land");
  auto cloud_class = node.empty() ? -1 : get_class_value(node, "cloud");
  if (sea_class < 0 || land_class < 0 || cloud_class < 0) {
    spdlog::warn("Could not read class values from kernel node {}, using the "
                 "kernel's (cloud 0, land 1, sea 2)",
                 node.string());
    label_values = kernel_label_values;
  } else {
    label_values = {static_cast<uint8_t>(sea_class),
                    static_cast<uint8_t>(land_class),
                    static_cast<uint8_t>(cloud_class)};
  }
  spdlog::info("KernelDevice {} initialized, built for {} MHz. Its classes "
               "are: sea={}, land={}, cloud={}",
               KERNEL_NAME, KERNEL_CLOCK_MHZ, label_values.sea,
               label_values.land, label_values.cloud);
}

KernelDevice::~KernelDevice() { kernelRelease(&device); }

uint64_t KernelDevice::run(uint64_t cube_address, uint64_t scores_address,
                           uint32_t height, uint32_t width,
                           std::chrono::seconds timeout) {
  if (!isIdle()) {
    throw std::runtime_error("The kernel is not idle");
  }
  kernelSetInputs(&device, cube_address); // every input port reads the cube
  kernelSetOutput(&device, scores_address);
  kernelSetSize(&device, height, width);
  spdlog::debug("Start: {} x {} pixels, cube at 0x{:x}, scores at 0x{:x}",
                height, width, cube_address, scores_address);
  const auto start = std::chrono::steady_clock::now();
  kernelStart(&device);
  // Polled without sleeping: a start takes milliseconds, a sleep would add to it.
  while (!kernelIsDone(&device)) {
    if (std::chrono::steady_clock::now() - start > timeout) {
      throw std::runtime_error("Timeout waiting for the kernel to finish");
    }
  }
  return kernelCycles(&device);
}

bool KernelDevice::isReady() { return kernelIsReady(&device); }

bool KernelDevice::isIdle() { return kernelIsIdle(&device); }
