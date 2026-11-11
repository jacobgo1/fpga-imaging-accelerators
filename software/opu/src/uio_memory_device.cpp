#include "uio_memory_device.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <regex>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

std::filesystem::path uioDeviceDir = "/dev";
std::filesystem::path uioClassDir = "/sys/class/uio";

int get_device_index_by_filename(std::string subsystem_name,
                                 std::string device_filename) {
  std::regex index_regex(subsystem_name + "(\\d+)");
  std::smatch match;
  if (std::regex_search(device_filename, match, index_regex)) {
    return std::stoi(match[1].str());
  }
  return -1;
}

int get_sysfs_by_name(std::filesystem::path subsystem_dir,
                      std::string subsystem_name, std::string device_name) {
  for (auto const &dir_entry :
       std::filesystem::directory_iterator{subsystem_dir}) {
    if (!dir_entry.is_directory()) {
      continue;
    }

    const auto name_path = dir_entry.path() / "name";
    if (!std::filesystem::exists(name_path)) {
      continue;
    }

    std::ifstream name_file(name_path);
    if (!name_file) {
      spdlog::warn("Could not open name file for {}: {}",
                   dir_entry.path().string(), strerror(errno));
      continue;
    }

    std::string name_buffer;
    std::getline(name_file, name_buffer);
    if (name_buffer == device_name) {
      auto device_filename = dir_entry.path().stem().string();
      auto device_index =
          get_device_index_by_filename(subsystem_name, device_filename);
      if (device_index >= 0) {
        return device_index;
      } else {
        spdlog::warn(
            "Found {} device with matching name '{}' but could not parse "
            "index from filename '{}'",
            subsystem_name, device_name, device_filename);
      }
    }
  }
  return -1;
}

uint64_t read_sysfs_hex(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Could not open " + path.string());
  }

  std::string line;
  std::getline(file, line);
  try {
    return static_cast<uint64_t>(stoull(line, nullptr, 16));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to parse hex value from " + path.string() +
                             ": " + e.what());
  }
}

UioMap::UioMap(std::filesystem::path uio_device_path,
               std::filesystem::path map_dir)
    : device_path(uio_device_path) {
  std::filesystem::path name_path = map_dir / "name";
  std::filesystem::path addr_path = map_dir / "addr";
  std::filesystem::path size_path = map_dir / "size";
  std::filesystem::path offset_path = map_dir / "offset";

  if (!std::filesystem::exists(addr_path) ||
      !std::filesystem::exists(size_path) ||
      !std::filesystem::exists(offset_path)) {
    throw std::runtime_error("No valid map found in " + map_dir.string());
  }

  if (std::filesystem::exists(name_path)) {
    std::ifstream name_file(name_path);
    if (name_file) {
      ;
      std::getline(name_file, map_name);
    } else {
      spdlog::warn("Could not open file {}: {}", name_path.string(),
                   strerror(errno));
    }
  }

  map_index = get_device_index_by_filename("map", map_dir.filename().string());
  if (map_index < 0) {
    throw std::runtime_error("Could not parse map index from " +
                             map_dir.filename().string());
  }
  map_address = read_sysfs_hex(addr_path);
  map_size = read_sysfs_hex(size_path);
  map_offset = read_sysfs_hex(offset_path);

  spdlog::info("Initialized UIO map {} with address 0x{:x}, size {}, offset "
               "{}, name '{}'",
               map_index, map_address, map_size, map_offset, map_name);
}

UioMap::~UioMap() { unmapMemory(); }

void UioMap::mapMemory(size_t size) {
  unmapMemory();

  if (size > map_size) {
    throw std::runtime_error("Requested map size " + std::to_string(size) +
                             " exceeds available size " +
                             std::to_string(map_size));
  }

  fd = open(device_path.c_str(), O_RDWR | O_SYNC);
  if (fd < 0) {
    throw std::runtime_error("Could not open " + device_path.string() + ": " +
                             strerror(errno));
  }

  buffer_size = size + map_offset;
  buffer_ptr = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    fd, map_index * getpagesize());
  if (buffer_ptr == MAP_FAILED) {
    std::string error_str = "Could not mmap " + std::to_string(buffer_size) +
                            " bytes to uio device " + device_path.string() +
                            " map " + std::to_string(map_index) + ": " +
                            std::string(strerror(errno));
    close(fd);
    fd = -1;
    buffer_size = 0;
    buffer_ptr = nullptr;
    throw std::runtime_error(error_str);
  }
}

void UioMap::unmapMemory() {
  if (buffer_ptr != nullptr && buffer_size != 0) {
    munmap(buffer_ptr, buffer_size);
  }
  buffer_ptr = nullptr;
  buffer_size = 0;

  if (fd >= 0) {
    close(fd);
    fd = -1;
  }
}

void UioMap::dump(std::filesystem::path dump_path) {
  if (buffer_ptr == nullptr || buffer_size == 0) {
    spdlog::error("Map is not currently mapped, cannot dump contents");
    return;
  }
  std::ofstream dump_file(dump_path, std::ios::binary);
  if (!dump_file) {
    spdlog::error("Failed to write dump file: {}: {}", dump_path.string(),
                  std::string(strerror(errno)));
    return;
  }
  dump_file.write(reinterpret_cast<const char *>(data()), size());
  if (!dump_file) {
    spdlog::error("Failed to write dump file: {}: {}", dump_path.string(),
                  std::string(strerror(errno)));
    return;
  }
  dump_file.close();
  spdlog::info("Dumped {} bytes of map {} to {}", size(), name(),
               dump_path.string());
}

UioDevice::UioDevice(std::string name) : uio_name(name) {
  uio_index = get_sysfs_by_name(uioClassDir, "uio", uio_name);
  if (uio_index < 0) {
    throw std::runtime_error("Could not find UIO device with name: " +
                             uio_name);
  }

  // Find all maps in uio
  maps_dir = uioClassDir / ("uio" + std::to_string(uio_index)) / "maps";
  if (!std::filesystem::exists(maps_dir) ||
      !std::filesystem::is_directory(maps_dir)) {
    throw std::runtime_error("Maps directory not found for uio" +
                             std::to_string(uio_index));
  }
  for (auto const &dir_entry : std::filesystem::directory_iterator{maps_dir}) {
    if (!dir_entry.is_directory()) {
      continue;
    }
    int device_index = get_device_index_by_filename(
        "map", dir_entry.path().filename().string());
    if (device_index >= 0) {
      n_maps++;
    }
  }
  spdlog::info("Initialized UIO device {} with name {} and {} maps", uio_index,
               uio_name, n_maps);
}

UioMap UioDevice::getMap(int map_index) {
  if (map_index < 0 || static_cast<size_t>(map_index) >= n_maps) {
    throw std::runtime_error(
        "Found no map with index " + std::to_string(map_index) +
        " for uio with index " + std::to_string(uio_index));
  }
  auto device_path = uioDeviceDir / ("uio" + std::to_string(uio_index));
  auto map_dir = maps_dir / ("map" + std::to_string(map_index));
  return UioMap(device_path, map_dir);
}
