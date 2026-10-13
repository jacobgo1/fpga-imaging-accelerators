#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

// A piece of physical address space mapped into this process: the kernel's
// registers, or the DDR the kernel reads the cube from and writes scores to.
// Three ways to get one:
//   uio     a UIO device from the device tree (HYPSO-2, Petalinux), by name
//   devmem  /dev/mem at a physical address (PYNQ, whose device tree has no UIO)
//   heap    ordinary memory standing in for DDR (the --sim build, on a PC)
// uio and devmem mappings are uncached, so the CPU and the kernel always see
// the same bytes and nothing has to be flushed.
class Region {
public:
  static Region uio(const std::string &uio_name, int map_index);
  static Region devmem(uint64_t phys, size_t size);
  static Region heap(uint64_t phys, size_t size);

  Region(Region &&other) noexcept;
  Region &operator=(Region &&other) = delete;
  Region(const Region &) = delete;
  ~Region();

  uint64_t phys() const { return phys_; }
  size_t size() const { return size_; }
  volatile uint8_t *data() const { return data_; }
  const std::string &name() const { return name_; }

private:
  Region() = default;
  std::string name_;
  uint64_t phys_ = 0;
  size_t size_ = 0;
  volatile uint8_t *data_ = nullptr;
  void *mapping_ = nullptr;   // what munmap or free gets back
  size_t mapping_size_ = 0;
  int fd_ = -1;
  bool heap_ = false;
};

// Copies to and from a Region. Uncached memory takes aligned accesses only:
// glibc's memcpy uses unaligned loads and DC ZVA, which fault there on arm64
// (SIGBUS). Offsets and sizes must be multiples of 8.
void copy_to(Region &region, size_t offset, const void *src, size_t bytes);
void copy_from(const Region &region, size_t offset, void *dst, size_t bytes);

// The kernel's control registers, as the driver sees them.
class Registers {
public:
  virtual ~Registers() = default;
  virtual uint32_t read(uint32_t offset) = 0;
  virtual void write(uint32_t offset, uint32_t value) = 0;
};

class MmioRegisters : public Registers {
public:
  explicit MmioRegisters(Region region) : region_(std::move(region)) {}
  uint32_t read(uint32_t offset) override;
  void write(uint32_t offset, uint32_t value) override;
  const Region &region() const { return region_; }

private:
  Region region_;
};
