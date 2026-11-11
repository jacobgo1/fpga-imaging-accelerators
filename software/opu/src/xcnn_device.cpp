#include "xcnn_device.h"
#include <byteswap.h>
#include <fstream>
#include <spdlog/spdlog.h>

template <typename Tin, typename Tout> Tout direct_conversion(Tin input) {
  return reinterpret_cast<Tout &>(input);
}

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

const char *xcnn_strerror(int errcode) {
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

CnnDevice::CnnDevice(CaptureConfig &capture_config, AddressConfig &addresses,
                     std::string uio_name, std::string compatible) {
  int ret = kernelInitialize(&device, uio_name.c_str());
  if (ret != XST_SUCCESS) {
    throw std::runtime_error(std::string(KERNEL_NAME) + " initialize failed: " +
                             std::string(xcnn_strerror(ret)));
  }
  // Every input port of the kernel reads the cube; its output is the scores
  kernelSetInputs(&device, addresses.dn_address);
  kernelSetOutput(&device, addresses.output_address);
  kernelSetSize(&device, capture_config.frame_count, capture_config.row_count);

  auto cnn_node = find_compatible_node(compatible);
  auto sea_class = get_class_value(cnn_node, "sea");
  auto land_class = get_class_value(cnn_node, "land");
  auto cloud_class = get_class_value(cnn_node, "cloud");
  if (sea_class < 0 || land_class < 0 || cloud_class < 0) {
    spdlog::warn("Could not read class values from CNN node {}, using default "
                 "label values",
                 cnn_node.string());
    label_values = default_label_values;
  } else {
    label_values = {static_cast<uint8_t>(sea_class),
                    static_cast<uint8_t>(land_class),
                    static_cast<uint8_t>(cloud_class)};
  }

  spdlog::info("CnnDevice initialized for {} (built for {} MHz): width={}, "
               "height={}, dn_address=0x{:x}, output_address=0x{:x}. Labels "
               "are: sea={}, land={}, cloud={}",
               KERNEL_NAME, KERNEL_CLOCK_MHZ, capture_config.row_count,
               capture_config.frame_count, addresses.dn_address,
               addresses.output_address, label_values.sea, label_values.land,
               label_values.cloud);
}

CnnDevice::~CnnDevice() { kernelRelease(&device); }

void CnnDevice::start() { kernelStart(&device); }

bool CnnDevice::isReady() { return kernelIsReady(&device); }

bool CnnDevice::isDone() { return kernelIsDone(&device); }

bool CnnDevice::isIdle() { return kernelIsIdle(&device); }

// The kernel's own count of its cycles in the last run, 0 if it keeps none
uint64_t CnnDevice::cycles() { return kernelCycles(&device); }
