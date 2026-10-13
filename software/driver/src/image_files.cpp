#include "image_files.hpp"

#include "formats.hpp"

#include <cstring>
#include <sys/types.h>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace {

std::string shape_text(const std::vector<size_t> &shape) {
  std::string s = "(";
  for (size_t i = 0; i < shape.size(); i++) {
    s += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? ", " : "");
  }
  return s + ")";
}

bool ends_with(const std::string &s, const std::string &end) {
  return s.size() >= end.size() && s.compare(s.size() - end.size(), end.size(), end) == 0;
}

}  // namespace

NpyArray read_npy_header(std::FILE *file, const std::string &path) {
  // The .npy format: magic, version, header length, then a Python dict literal.
  unsigned char start[10];
  if (std::fread(start, 1, 10, file) != 10 || std::memcmp(start, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error(path + " is not a .npy file");
  }
  size_t header_len = start[8] | (start[9] << 8);
  long offset = 10;
  if (start[6] >= 2) {   // versions 2 and 3: a 4-byte length
    unsigned char more[2];
    if (std::fread(more, 1, 2, file) != 2) throw std::runtime_error(path + ": short header");
    header_len |= (static_cast<size_t>(more[0]) << 16) | (static_cast<size_t>(more[1]) << 24);
    offset = 12;
  }
  std::string header(header_len, '\0');
  if (std::fread(header.data(), 1, header_len, file) != header_len) {
    throw std::runtime_error(path + ": short header");
  }

  NpyArray array;
  array.data_offset = offset + static_cast<long>(header_len);
  std::smatch m;
  if (!std::regex_search(header, m, std::regex(R"('descr'\s*:\s*'([^']+)')"))) {
    throw std::runtime_error(path + ": no descr in the .npy header");
  }
  array.descr = m[1];
  if (std::regex_search(header, std::regex(R"('fortran_order'\s*:\s*True)"))) {
    throw std::runtime_error(path + " is in Fortran order; save it with np.ascontiguousarray");
  }
  if (!std::regex_search(header, m, std::regex(R"('shape'\s*:\s*\(([^)]*)\))"))) {
    throw std::runtime_error(path + ": no shape in the .npy header");
  }
  const std::string dims = m[1];
  const std::regex number(R"(\d+)");
  for (auto it = std::sregex_iterator(dims.begin(), dims.end(), number); it != std::sregex_iterator(); ++it) {
    array.shape.push_back(std::stoull(it->str()));
  }
  return array;
}

void write_npy(const std::string &path, const std::string &descr, const std::vector<size_t> &shape,
               const void *data, size_t bytes) {
  std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
                       shape_text(shape) + ", }";
  // Version 1.0; the values start at a multiple of 64 bytes, as numpy writes it.
  const size_t total = (10 + header.size() + 1 + 63) / 64 * 64;
  header.append(total - 10 - header.size() - 1, ' ');
  header += '\n';
  std::ofstream file(path, std::ios::binary);
  const uint16_t len = static_cast<uint16_t>(header.size());
  file.write("\x93NUMPY\x01\x00", 8);
  file.put(static_cast<char>(len & 0xff));
  file.put(static_cast<char>(len >> 8));
  file << header;
  file.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
  if (!file) {
    throw std::runtime_error("cannot write " + path);
  }
}

void write_array(const std::string &path, const std::string &descr, const std::vector<size_t> &shape,
                 const void *data, size_t bytes) {
  if (ends_with(path, ".npy")) {
    write_npy(path, descr, shape, data, bytes);
    return;
  }
  std::ofstream file(path, std::ios::binary);
  file.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
  if (!file) {
    throw std::runtime_error("cannot write " + path);
  }
}

void read_capture_config(const std::string &path, uint32_t &height, uint32_t &width) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot read " + path);
  }
  const std::regex setting(R"(^\s*(\w+)\s*=\s*"?([^";\s]+))");
  std::string line;
  std::smatch m;
  bool have_height = false, have_width = false;
  while (std::getline(file, line)) {
    if (!std::regex_search(line, m, setting)) continue;
    if (m[1] == "frame_count") {
      height = static_cast<uint32_t>(std::stoul(m[2]));
      have_height = true;
    } else if (m[1] == "row_count") {
      width = static_cast<uint32_t>(std::stoul(m[2]));
      have_width = true;
    }
  }
  if (!have_height || !have_width) {
    throw std::runtime_error(path + " needs frame_count and row_count");
  }
}

CubeFile::CubeFile(const std::string &path, uint32_t height, uint32_t width) : path_(path) {
  file_ = std::fopen(path.c_str(), "rb");
  if (!file_) {
    throw std::runtime_error("cannot open the cube " + path);
  }
  if (ends_with(path, ".npy")) {
    NpyArray a = read_npy_header(file_, path);
    if (a.descr != "<u2" || a.shape.size() != 3 || a.shape[2] != CUBE_BANDS) {
      throw std::runtime_error(path + " must be uint16 (H, W, 120), not " + a.descr + " " +
                               shape_text(a.shape));
    }
    height_ = static_cast<uint32_t>(a.shape[0]);
    width_ = static_cast<uint32_t>(a.shape[1]);
    data_offset_ = a.data_offset;
  } else {
    if (height == 0 || width == 0) {
      throw std::runtime_error("a raw cube needs its size: --capture-config, or --height and --width");
    }
    height_ = height;
    width_ = width;
  }
  // fseeko/ftello with _FILE_OFFSET_BITS=64: long is 32 bits on the Zynq-7030.
  fseeko(file_, 0, SEEK_END);
  const off_t size = ftello(file_);
  const off_t needed = data_offset_ + static_cast<off_t>(height_) * static_cast<off_t>(row_bytes());
  if (size < needed) {
    throw std::runtime_error(path + " has " + std::to_string(size) + " bytes; " +
                             std::to_string(height_) + " x " + std::to_string(width_) +
                             " x 120 uint16 needs " + std::to_string(needed));
  }
}

CubeFile::~CubeFile() {
  if (file_) std::fclose(file_);
}

size_t CubeFile::row_bytes() const { return size_t(width_) * IN_PIXEL; }

void CubeFile::read_rows(uint32_t first, uint32_t count, void *dst) {
  fseeko(file_, data_offset_ + static_cast<off_t>(first) * static_cast<off_t>(row_bytes()), SEEK_SET);
  if (std::fread(dst, row_bytes(), count, file_) != count) {
    throw std::runtime_error("short read from " + path_);
  }
}

std::vector<int32_t> read_scores_npy(const std::string &path, uint32_t height, uint32_t width) {
  std::FILE *file = std::fopen(path.c_str(), "rb");
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  try {
    NpyArray a = read_npy_header(file, path);
    const std::vector<size_t> want = {height, width, 4};
    if (a.descr != "<i4" || a.shape != want) {
      throw std::runtime_error(path + " must be int32 " + shape_text(want) + ", not " + a.descr + " " +
                               shape_text(a.shape));
    }
    std::vector<int32_t> values(size_t(height) * width * 4);
    if (std::fread(values.data(), sizeof(int32_t), values.size(), file) != values.size()) {
      throw std::runtime_error("short read from " + path);
    }
    std::fclose(file);
    return values;
  } catch (...) {
    std::fclose(file);
    throw;
  }
}
