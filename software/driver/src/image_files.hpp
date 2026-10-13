#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// The files the driver reads and writes. Everything is little endian, as on
// the ARM, so a .npy from the PC is used as it is.
//   cube:   .npy (uint16, H x W x 120, from tools/justounetsimple_image.py), or
//           raw BIP as the HYPSO payload writes it (frames x rows x 120 uint16;
//           frames = height, rows = width)
//   scores: what the kernel wrote, H x W x 4 int32 (.npy, or raw otherwise)
//   labels: one class per pixel, uint8 (.npy, or raw otherwise)

struct NpyArray {
  std::string descr;            // '<u2', '<i4', '|u1'
  std::vector<size_t> shape;
  long data_offset = 0;         // where the values start in the file
};

NpyArray read_npy_header(std::FILE *file, const std::string &path);
void write_npy(const std::string &path, const std::string &descr, const std::vector<size_t> &shape,
               const void *data, size_t bytes);
// .npy when the name ends in .npy, raw bytes otherwise.
void write_array(const std::string &path, const std::string &descr, const std::vector<size_t> &shape,
                 const void *data, size_t bytes);

// HYPSO's capture_config.ini: frame_count (height) and row_count (width).
void read_capture_config(const std::string &path, uint32_t &height, uint32_t &width);

// The raw cube, read a strip of rows at a time so that only one strip is ever
// in the ARM's memory.
class CubeFile {
public:
  // height/width are needed for raw files only (0: take them from the .npy).
  CubeFile(const std::string &path, uint32_t height, uint32_t width);
  ~CubeFile();
  CubeFile(const CubeFile &) = delete;
  CubeFile &operator=(const CubeFile &) = delete;

  uint32_t height() const { return height_; }
  uint32_t width() const { return width_; }
  size_t row_bytes() const;
  void read_rows(uint32_t first, uint32_t count, void *dst);

private:
  std::string path_;
  std::FILE *file_ = nullptr;
  long data_offset_ = 0;
  uint32_t height_ = 0, width_ = 0;
};

// The whole of an int32 H x W x 4 .npy (reference_fixed.npy).
std::vector<int32_t> read_scores_npy(const std::string &path, uint32_t height, uint32_t width);
