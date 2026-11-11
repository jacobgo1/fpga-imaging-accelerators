#pragma once
// The kernel's work on a whole capture, without the board's devices: the cube from its
// file into the memory the kernel reads, in strips if it does not fit, the kernel's
// scores out as HYPSO's labels (and, if asked, the scores themselves), and the classes
// checked against true labels. Only the C++ standard library, so the tests run it on
// the PC with the kernel's C++ as the board (tests/test_opu_driver.py).

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iosfwd>
#include <optional>

constexpr int BANDS = 120; // raw bands a pixel in a capture
constexpr int LANES = 4;   // the kernel's int32s a pixel: three scores * 2^OUT_FRAC, then the class
constexpr int PATCH = 32;  // strips are a multiple of it: the 2D kernels' patches are independent

// What the labels file holds for each class (cnn-1d-picobob's LabelValues).
struct LabelValues {
  uint8_t sea;
  uint8_t land;
  uint8_t cloud;
  bool operator==(const LabelValues &other) const {
    return sea == other.sea && land == other.land && cloud == other.cloud;
  }
};
// HYPSO's, in the labels file; the kernel's own (0 cloud, 1 land, 2 sea), which the
// devicetree overlay also states.
constexpr LabelValues default_label_values = {0, 1, 2};
constexpr LabelValues kernel_label_values = {2, 1, 0};

// A raw cube file: H x W x 120 uint16, band-interleaved by pixel. A .npy says its own
// size; a .bip (binned_cube.bip) has none, so height and width come from the capture.
struct Cube {
  std::ifstream file; // at the start of the data
  long height = 0;    // frames (lines)
  long width = 0;     // pixels a frame (the capture's row count)
};
Cube openCube(const std::filesystem::path &path, long height = 0,
              long width = 0);

// Memory the kernel reaches: where the ARM sees it, and where the kernel does.
struct Buffer {
  uint8_t *data;
  uint64_t physical;
  size_t size;
};

// Starts the kernel on HEIGHT x WIDTH pixels at CUBE, its scores to SCORES (physical
// addresses), and waits for it: its own count of its cycles, 0 if it keeps none.
using StartKernel = std::function<uint64_t(
    uint64_t cube, uint64_t scores, uint32_t height, uint32_t width)>;

struct Outputs {
  std::filesystem::path labels;                // H x W uint8, HYPSO's values
  std::optional<std::filesystem::path> scores; // H x W x 4 int32 .npy, what the kernel wrote
  std::optional<std::filesystem::path> truth;  // H x W integer .npy: 0 cloud, 1 land, 2 sea
  LabelValues kernel_values = kernel_label_values;
};

struct Result {
  long height = 0, width = 0, rows = 0, starts = 0, patches = 0;
  double load_s = 0, fpga_s = 0, save_s = 0;
  uint64_t cycles = 0;
  bool has_cycles = false;
  // Against the true labels: the labelled pixels, those classified as labelled, and per
  // class the pixels both call it and either calls it (IoU), as the notebooks count them.
  bool checked = false;
  long labelled = 0, correct = 0, both[3] = {0, 0, 0}, either[3] = {0, 0, 0};
};

Result classify(Cube &cube, const Buffer &cube_memory,
                const Buffer &scores_memory, const StartKernel &start,
                const Outputs &outputs);

// The result as lines of text, the ones benchmark.sh reads.
void printResult(const Result &result, const Outputs &outputs,
                 std::ostream &out);
