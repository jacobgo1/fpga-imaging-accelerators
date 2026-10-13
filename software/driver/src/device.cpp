#include "device.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

std::string read_line(const fs::path &path) {
  std::ifstream file(path);
  std::string line;
  if (!file || !std::getline(file, line)) {
    throw std::runtime_error("cannot read " + path.string());
  }
  return line;
}

uint64_t read_hex(const fs::path &path) { return std::stoull(read_line(path), nullptr, 16); }

std::string errno_text() { return std::strerror(errno); }

}  // namespace

Region Region::uio(const std::string &uio_name, int map_index) {
  // /sys/class/uio/uioN/name is the device tree's linux,uio-name.
  fs::path dir;
  std::error_code error;
  for (const auto &entry : fs::directory_iterator("/sys/class/uio", error)) {
    if (fs::exists(entry.path() / "name") && read_line(entry.path() / "name") == uio_name) {
      dir = entry.path();
      break;
    }
  }
  if (dir.empty()) {
    throw std::runtime_error("no UIO device named '" + uio_name +
                             "': is the device tree overlay loaded (firmware/*.dtsi)?");
  }
  const fs::path map = dir / "maps" / ("map" + std::to_string(map_index));
  if (!fs::exists(map / "addr")) {
    throw std::runtime_error("UIO device '" + uio_name + "' has no map" + std::to_string(map_index));
  }

  Region r;
  r.name_ = uio_name;
  r.phys_ = read_hex(map / "addr");
  r.size_ = read_hex(map / "size");
  // "offset": where the map starts within its first page (0 when page aligned).
  const size_t in_page = fs::exists(map / "offset") ? read_hex(map / "offset") : 0;
  const std::string device = "/dev/" + dir.filename().string();
  r.fd_ = open(device.c_str(), O_RDWR | O_SYNC);
  if (r.fd_ < 0) {
    throw std::runtime_error("cannot open " + device + ": " + errno_text());
  }
  // UIO selects map N by an mmap offset of N pages.
  r.mapping_size_ = r.size_ + in_page;
  r.mapping_ = mmap(nullptr, r.mapping_size_, PROT_READ | PROT_WRITE, MAP_SHARED, r.fd_,
                    static_cast<off_t>(map_index) * getpagesize());
  if (r.mapping_ == MAP_FAILED) {
    r.mapping_ = nullptr;
    throw std::runtime_error("cannot mmap " + device + " map" + std::to_string(map_index) + ": " +
                             errno_text());
  }
  r.data_ = static_cast<volatile uint8_t *>(r.mapping_) + in_page;
  return r;
}

Region Region::devmem(uint64_t phys, size_t size) {
  Region r;
  r.name_ = "/dev/mem";
  r.phys_ = phys;
  r.size_ = size;
  const uint64_t page = static_cast<uint64_t>(getpagesize());
  const uint64_t base = phys & ~(page - 1);
  // O_SYNC: an uncached mapping, as with UIO.
  r.fd_ = open("/dev/mem", O_RDWR | O_SYNC);
  if (r.fd_ < 0) {
    throw std::runtime_error("cannot open /dev/mem (run as root): " + errno_text());
  }
  r.mapping_size_ = size + (phys - base);
  r.mapping_ = mmap(nullptr, r.mapping_size_, PROT_READ | PROT_WRITE, MAP_SHARED, r.fd_,
                    static_cast<off_t>(base));
  if (r.mapping_ == MAP_FAILED) {
    r.mapping_ = nullptr;
    throw std::runtime_error("cannot mmap /dev/mem at physical address " + std::to_string(phys) +
                             ": " + errno_text());
  }
  r.data_ = static_cast<volatile uint8_t *>(r.mapping_) + (phys - base);
  return r;
}

Region Region::heap(uint64_t phys, size_t size) {
  Region r;
  r.name_ = "heap";
  r.phys_ = phys;
  r.size_ = size;
  r.heap_ = true;
  r.mapping_ = std::calloc(size, 1);
  if (!r.mapping_) {
    throw std::runtime_error("out of memory for a " + std::to_string(size) + " byte buffer");
  }
  r.data_ = static_cast<volatile uint8_t *>(r.mapping_);
  return r;
}

Region::Region(Region &&other) noexcept
    : name_(std::move(other.name_)), phys_(other.phys_), size_(other.size_), data_(other.data_),
      mapping_(other.mapping_), mapping_size_(other.mapping_size_), fd_(other.fd_),
      heap_(other.heap_) {
  other.mapping_ = nullptr;
  other.data_ = nullptr;
  other.fd_ = -1;
}

Region::~Region() {
  if (mapping_) {
    if (heap_) {
      std::free(mapping_);
    } else {
      munmap(mapping_, mapping_size_);
    }
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

static void check_copy(const Region &region, size_t offset, size_t bytes) {
  if (offset % 8 || bytes % 8) {
    throw std::logic_error("copies to a Region must be 8-byte aligned");
  }
  if (offset + bytes > region.size()) {
    throw std::logic_error("copy past the end of " + region.name());
  }
}

void copy_to(Region &region, size_t offset, const void *src, size_t bytes) {
  check_copy(region, offset, bytes);
  auto *to = reinterpret_cast<volatile uint64_t *>(region.data() + offset);
  const auto *from = static_cast<const uint8_t *>(src);
  for (size_t i = 0; i < bytes / 8; i++) {
    uint64_t word;
    std::memcpy(&word, from + 8 * i, 8);   // the source may be unaligned; it is cached memory
    to[i] = word;
  }
}

void copy_from(const Region &region, size_t offset, void *dst, size_t bytes) {
  check_copy(region, offset, bytes);
  const auto *from = reinterpret_cast<const volatile uint64_t *>(region.data() + offset);
  auto *to = static_cast<uint8_t *>(dst);
  for (size_t i = 0; i < bytes / 8; i++) {
    const uint64_t word = from[i];
    std::memcpy(to + 8 * i, &word, 8);
  }
}

uint32_t MmioRegisters::read(uint32_t offset) {
  return *reinterpret_cast<volatile uint32_t *>(region_.data() + offset);
}

void MmioRegisters::write(uint32_t offset, uint32_t value) {
  *reinterpret_cast<volatile uint32_t *>(region_.data() + offset) = value;
}
