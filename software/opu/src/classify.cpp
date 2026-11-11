#include "classify.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double secondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

// A .npy file's header dict, leaving the file at the start of the data.
std::string npyHeader(std::ifstream &file, const std::filesystem::path &path) {
  unsigned char start[12];
  if (!file.read(reinterpret_cast<char *>(start), 10) ||
      std::memcmp(start, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error(path.string() + ": not a .npy file");
  }
  size_t n = start[8] | static_cast<size_t>(start[9]) << 8;
  if (start[6] >= 2) { // versions 2 and 3: a 4-byte length
    if (!file.read(reinterpret_cast<char *>(start + 10), 2)) {
      throw std::runtime_error(path.string() + ": a .npy file cut short");
    }
    n |= static_cast<size_t>(start[10]) << 16 | static_cast<size_t>(start[11]) << 24;
  }
  std::string header(n, '\0');
  if (!file.read(header.data(), static_cast<std::streamsize>(n))) {
    throw std::runtime_error(path.string() + ": a .npy file cut short");
  }
  if (header.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error(path.string() + ": expected C order, the header is " + header);
  }
  return header;
}

// The header's shape, exactly DIMS sizes.
std::vector<long> npyShape(const std::string &header, size_t dims,
                           const std::filesystem::path &path) {
  const std::string key = "'shape': (";
  auto at = header.find(key);
  if (at == std::string::npos) {
    throw std::runtime_error(path.string() + ": a .npy header without a shape");
  }
  const char *p = header.c_str() + at + key.size();
  std::vector<long> shape;
  while (*p != ')') {
    char *end;
    long v = std::strtol(p, &end, 10);
    if (end == p) {
      throw std::runtime_error(path.string() + ": cannot read the shape in " + header);
    }
    shape.push_back(v);
    for (p = end; *p == ',' || *p == ' '; p++) {
    }
  }
  if (shape.size() != dims) {
    throw std::runtime_error(path.string() + ": expected " + std::to_string(dims) +
                             " dimensions, the header is " + header);
  }
  return shape;
}

// True labels of any integer type (tools/image.py writes int64, a capture's own are uint8).
struct Truth {
  std::ifstream file;
  int bytes = 0;
  bool is_signed = false;
};

Truth openTruth(const std::filesystem::path &path, long height, long width) {
  Truth truth;
  truth.file.open(path, std::ios::binary);
  if (!truth.file) {
    throw std::runtime_error("cannot open the labels " + path.string());
  }
  auto header = npyHeader(truth.file, path);
  const std::string key = "'descr': '";
  auto at = header.find(key);
  const char *d = at == std::string::npos ? "" : header.c_str() + at + key.size();
  if ((d[0] != '<' && d[0] != '|') || (d[1] != 'i' && d[1] != 'u') ||
      !std::strchr("1248", d[2]) || !d[2] || d[3] != '\'') {
    throw std::runtime_error(path.string() + ": expected integer labels, the header is " + header);
  }
  truth.bytes = d[2] - '0';
  truth.is_signed = d[1] == 'i';
  auto shape = npyShape(header, 2, path);
  if (shape[0] != height || shape[1] != width) {
    throw std::runtime_error(path.string() + ": the labels are not H x W for this cube");
  }
  return truth;
}

// A little-endian label of that type, as a number.
int64_t labelAt(const unsigned char *p, int bytes, bool is_signed) {
  uint64_t v = 0;
  for (int i = bytes - 1; i >= 0; i--) {
    v = v << 8 | p[i];
  }
  if (is_signed && bytes < 8 && ((v >> (8 * bytes - 1)) & 1)) {
    v |= ~uint64_t{0} << (8 * bytes);
  }
  return static_cast<int64_t>(v);
}

// The header of an int32 H x W x LANES .npy file: version 1, padded to 64 bytes.
void writeNpyHeader(std::ofstream &file, long height, long width) {
  std::string dict = "{'descr': '<i4', 'fortran_order': False, 'shape': (" +
                     std::to_string(height) + ", " + std::to_string(width) + ", " +
                     std::to_string(LANES) + "), }";
  size_t len = (10 + dict.size() + 1 + 63) / 64 * 64 - 10; // the dict, spaces, a newline
  dict.resize(len - 1, ' ');
  dict += '\n';
  const unsigned char start[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
                                   static_cast<unsigned char>(len),
                                   static_cast<unsigned char>(len >> 8)};
  file.write(reinterpret_cast<const char *>(start), 10);
  file << dict;
}

// The kernel's class -> the labels file's value (cnn-1d-picobob's translation).
uint8_t labelValue(int32_t cls, const LabelValues &kernel) {
  if (cls == kernel.sea) {
    return default_label_values.sea;
  }
  if (cls == kernel.land) {
    return default_label_values.land;
  }
  return default_label_values.cloud;
}

} // namespace

Cube openCube(const std::filesystem::path &path, long height, long width) {
  Cube cube;
  cube.file.open(path, std::ios::binary);
  if (!cube.file) {
    throw std::runtime_error("cannot open the cube " + path.string());
  }
  if (path.extension() == ".npy") {
    auto header = npyHeader(cube.file, path);
    if (header.find("'descr': '<u2'") == std::string::npos) {
      throw std::runtime_error(path.string() + ": expected '<u2' (raw counts), the header is " + header);
    }
    auto shape = npyShape(header, 3, path);
    cube.height = shape[0];
    cube.width = shape[1];
    if (shape[2] != BANDS) {
      throw std::runtime_error(path.string() + ": the cube must have 120 bands");
    }
  } else {
    if (height <= 0 || width <= 0) {
      throw std::runtime_error(path.string() + ": a raw cube needs its size from the capture config");
    }
    cube.height = height;
    cube.width = width;
    const auto need = static_cast<uintmax_t>(height) * width * BANDS * 2;
    const auto have = std::filesystem::file_size(path);
    if (have < need) {
      throw std::runtime_error("Trying to read " + std::to_string(need) + " bytes from " +
                               path.string() + ", but it only contains " +
                               std::to_string(have) + " bytes");
    }
  }
  return cube;
}

Result classify(Cube &cube, const Buffer &cube_memory,
                const Buffer &scores_memory, const StartKernel &start,
                const Outputs &outputs) {
  Result result;
  const long h = cube.height, w = cube.width;
  result.height = h;
  result.width = w;

  // As many rows as fit, in and out, a multiple of 32 when the cube does not fit whole.
  const size_t in_row = static_cast<size_t>(w) * BANDS * 2;
  const size_t out_row = static_cast<size_t>(w) * LANES * 4;
  long rows = static_cast<long>(std::min(cube_memory.size / in_row, scores_memory.size / out_row));
  if (rows < h) {
    rows = rows / PATCH * PATCH;
    if (rows < PATCH) {
      throw std::runtime_error("the reserved memory is too small for 32 rows of this cube");
    }
  } else {
    rows = h;
  }
  result.rows = rows;

  std::ofstream labels(outputs.labels, std::ios::binary);
  if (!labels) {
    throw std::runtime_error("cannot write " + outputs.labels.string());
  }
  std::ofstream scores_file;
  if (outputs.scores) {
    scores_file.open(*outputs.scores, std::ios::binary);
    if (!scores_file) {
      throw std::runtime_error("cannot write " + outputs.scores->string());
    }
    writeNpyHeader(scores_file, h, w);
  }
  Truth truth;
  std::vector<unsigned char> truth_rows;
  if (outputs.truth) {
    truth = openTruth(*outputs.truth, h, w);
    truth_rows.resize(static_cast<size_t>(rows) * w * truth.bytes);
    result.checked = true;
  }
  std::vector<uint8_t> label_rows(static_cast<size_t>(rows) * w);

  for (long y = 0; y < h; y += rows) {
    const long n = std::min(rows, h - y);
    auto t = std::chrono::steady_clock::now();
    if (!cube.file.read(reinterpret_cast<char *>(cube_memory.data),
                        static_cast<std::streamsize>(n * in_row))) {
      throw std::runtime_error("the cube file is cut short");
    }
    result.load_s += secondsSince(t);

    t = std::chrono::steady_clock::now();
    const uint64_t cycles = start(cube_memory.physical, scores_memory.physical,
                                  static_cast<uint32_t>(n), static_cast<uint32_t>(w));
    result.fpga_s += secondsSince(t);
    result.cycles += cycles;

    t = std::chrono::steady_clock::now();
    const auto *scores = reinterpret_cast<const int32_t *>(scores_memory.data);
    const size_t pixels = static_cast<size_t>(n) * w;
    if (outputs.scores &&
        !scores_file.write(reinterpret_cast<const char *>(scores),
                           static_cast<std::streamsize>(n * out_row))) {
      throw std::runtime_error("cannot write the scores");
    }
    if (outputs.truth &&
        !truth.file.read(reinterpret_cast<char *>(truth_rows.data()),
                         static_cast<std::streamsize>(pixels * truth.bytes))) {
      throw std::runtime_error("the labels file is cut short");
    }
    for (size_t p = 0; p < pixels; p++) {
      const int32_t cls = scores[LANES * p + 3];
      label_rows[p] = labelValue(cls, outputs.kernel_values);
      if (!outputs.truth) {
        continue;
      }
      // The true labels are the kernel's classes (0 cloud, 1 land, 2 sea); others unlabelled.
      const int64_t label = labelAt(&truth_rows[p * truth.bytes], truth.bytes, truth.is_signed);
      if (label < 0 || label > 2) {
        continue;
      }
      result.labelled++;
      result.correct += cls == label;
      for (int c = 0; c < 3; c++) {
        result.both[c] += cls == c && label == c;
        result.either[c] += cls == c || label == c;
      }
    }
    if (!labels.write(reinterpret_cast<const char *>(label_rows.data()),
                      static_cast<std::streamsize>(pixels))) {
      throw std::runtime_error("cannot write " + outputs.labels.string());
    }
    result.save_s += secondsSince(t);
    result.patches += (n + PATCH - 1) / PATCH * ((w + PATCH - 1) / PATCH);
    result.starts++;
  }
  if (outputs.scores && !scores_file.flush()) {
    throw std::runtime_error("cannot write the scores");
  }
  if (!labels.flush()) {
    throw std::runtime_error("cannot write " + outputs.labels.string());
  }
  return result;
}

void printResult(const Result &r, const Outputs &outputs, std::ostream &out) {
  char line[256];
  std::snprintf(line, sizeof line, "%ld x %ld pixels, %ld patches, in %ld start%s of up to %ld rows\n",
                r.height, r.width, r.patches, r.starts, r.starts == 1 ? "" : "s", r.rows);
  out << line;
  std::snprintf(line, sizeof line, "cube into memory:   %8.3f s\n", r.load_s);
  out << line;
  std::snprintf(line, sizeof line, "FPGA:               %8.1f ms, %.1f us per patch\n",
                1e3 * r.fpga_s, 1e6 * r.fpga_s / static_cast<double>(r.patches));
  out << line;
  if (r.has_cycles) {
    out << "own count:          " << r.cycles << " cycles\n";
  }
  std::snprintf(line, sizeof line, "labels to the file: %8.3f s\n", r.save_s);
  out << line;
  if (!r.checked) {
    return;
  }
  std::snprintf(line, sizeof line, "accuracy against %s: %.2f%% of %ld labelled pixels\n",
                outputs.truth->string().c_str(),
                r.labelled ? 100.0 * r.correct / r.labelled : 0.0, r.labelled);
  out << line;
  const char *names[3] = {"cloud", "land", "sea"};
  out << "  IoU";
  for (int c = 0; c < 3; c++) {
    if (r.either[c]) {
      std::snprintf(line, sizeof line, "%s %s %.3f", c ? "," : "", names[c],
                    static_cast<double>(r.both[c]) / r.either[c]);
    } else { // neither has it: no IoU
      std::snprintf(line, sizeof line, "%s %s -", c ? "," : "", names[c]);
    }
    out << line;
  }
  out << "\n";
  // The counts behind those, for benchmark.sh to add up over images.
  out << "counts: " << r.labelled << " " << r.correct << " " << r.both[0] << " "
      << r.both[1] << " " << r.both[2] << " " << r.either[0] << " " << r.either[1]
      << " " << r.either[2]
      << " (labelled, correct, both cloud land sea, either cloud land sea)\n";
}
