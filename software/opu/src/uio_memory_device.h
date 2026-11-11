#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

class UioMap {
public:
  UioMap(std::filesystem::path uio_device_path, std::filesystem::path map_dir);
  ~UioMap();

  // Do not allow copy of this class
  UioMap(const UioMap &) = delete;
  UioMap &operator=(const UioMap &) = delete;

  void mapMemory(size_t size);
  void unmapMemory();
  void dump(std::filesystem::path dump_path);

  auto size() const { return buffer_size - map_offset; }
  auto data() const { return static_cast<char *>(buffer_ptr) + map_offset; }
  auto physicalAddress() const { return map_address; }
  auto maxSize() const { return map_size; }
  auto name() const { return map_name; }

private:
  void readMapInfo(std::filesystem::path map_dir);
  std::filesystem::path device_path;
  int fd = -1;
  void *buffer_ptr = nullptr;
  size_t buffer_size = 0;
  int map_index = -1;
  size_t map_size = 0;
  size_t map_offset = 0;
  uint64_t map_address = 0;
  std::string map_name;
};

class UioDevice {
public:
  UioDevice(std::string name);
  UioMap getMap(int map_index);
  auto getNumMaps() const { return n_maps; }
  auto name() const { return uio_name; }

private:
  std::string uio_name;
  int uio_index = -1;
  std::filesystem::path maps_dir;
  size_t n_maps = 0;
};
