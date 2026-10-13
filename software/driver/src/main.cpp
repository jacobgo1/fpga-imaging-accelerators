// jopt_driver: classifies a raw HYPSO capture with justounetsimple_opt on the
// FPGA, from the Linux on the board's ARM (Zynq-7030 on HYPSO-2, ZCU104 in the
// lab). What the notebook does, without Python: the cube goes from a file into
// DDR the kernel can reach, the kernel gets the addresses and sizes through its
// registers and classifies it, and the scores and classes go to files. If the
// reserved DDR is too small for the whole cube, it goes in strips of a multiple
// of 32 rows (whole patches, so the answer is the same), two strip buffers
// taking turns: the ARM copies strip s + 1 while the FPGA does strip s.

#include "accelerator.hpp"
#include "device.hpp"
#include "image_files.hpp"
#include "registers.hpp"
#ifdef JOPT_SIM
#include "sim.hpp"
#endif

#include "formats.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr size_t ALIGN = 4096;                                   // buffers start on a page
constexpr int EXIT_MISMATCH = 2;

const char *USAGE = R"(Usage: jopt_driver --input CUBE [options]

Classifies a raw HYPSO cube with justounetsimple_opt on the FPGA: 0 cloud, 1 land, 2 sea.

Input
  --input FILE           the raw cube: .npy (H, W, 120) uint16, or raw BIP as captured
  --capture-config FILE  HYPSO capture_config.ini: the BIP's frame_count (height) and row_count (width)
  --height N --width N   the size of a raw BIP cube, instead of --capture-config
Output
  --labels FILE          the class of every pixel, uint8 (.npy, or raw otherwise)
  --hypso-labels         write the labels in the HYPSO tools' order: 0 sea, 1 land, 2 cloud
  --scores FILE          what the kernel wrote: H x W x 4 int32, 3 scores * 2^16 and the class
  --reference FILE       reference_fixed.npy from the PC: check the output bit for bit (exit 2 if not)
Hardware
  --header FILE          the x<kernel>_hw.h of the bitstream's HLS build
                         (default: the one next to this program)
  --uio NAME             UIO name of the kernel's registers (default justounetsimple_opt)
  --uio-memory NAME      UIO name of the reserved DDR (default jopt_memory)
  --devmem-memory A:S    /dev/mem instead of UIO: reserved DDR at physical address A, S bytes
  --devmem-regs A        with --devmem-memory: the kernel's registers (default by --board)
  --board B              zcu104 (registers at 0xa0000000, the default), zynq7030 or zynq7020 (0x40000000)
  --strip-rows N         rows per start, a multiple of 32 (default: as many as fit)
  --timeout S            seconds per start before giving up (default 60)
)";

struct Options {
  std::string input, capture_config, labels, scores, reference, header;
  std::string uio = "justounetsimple_opt", uio_memory = "jopt_memory", board = "zcu104";
  std::optional<uint64_t> devmem_phys, devmem_regs;
  size_t devmem_size = 0;
  uint32_t height = 0, width = 0, strip_rows = 0;
  double timeout = 60;
  bool hypso_labels = false;
  size_t sim_memory = 64 << 20;
};

static uint64_t number(const std::string &text) { return std::stoull(text, nullptr, 0); }

[[maybe_unused]] static std::string hex(uint64_t value) {
  char text[24];
  std::snprintf(text, sizeof text, "0x%llx", static_cast<unsigned long long>(value));
  return text;
}

// The register header packed next to the program (scripts/package.sh):
// xjustounetsimple_opt_hw.h or xjustounetsimple_opt_zynq7030_hw.h.
static std::string find_header(const fs::path &dir) {
  std::vector<fs::path> found;
  std::error_code error;
  for (const auto &entry : fs::directory_iterator(dir.empty() ? "." : dir, error)) {
    const std::string name = entry.path().filename().string();
    if (name.size() > 5 && name[0] == 'x' && name.compare(name.size() - 5, 5, "_hw.h") == 0) {
      found.push_back(entry.path());
    }
  }
  if (found.size() != 1) {
    throw std::runtime_error(std::to_string(found.size()) + " x*_hw.h files next to the program: "
                             "say which register header with --header");
  }
  return found[0].string();
}

static Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(arg + " needs a value");
      return argv[++i];
    };
    if (arg == "-h" || arg == "--help") {
      std::cout << USAGE;
      std::exit(0);
    } else if (arg == "--input") o.input = next();
    else if (arg == "--capture-config") o.capture_config = next();
    else if (arg == "--height") o.height = static_cast<uint32_t>(number(next()));
    else if (arg == "--width") o.width = static_cast<uint32_t>(number(next()));
    else if (arg == "--labels") o.labels = next();
    else if (arg == "--hypso-labels") o.hypso_labels = true;
    else if (arg == "--scores") o.scores = next();
    else if (arg == "--reference") o.reference = next();
    else if (arg == "--header") o.header = next();
    else if (arg == "--uio") o.uio = next();
    else if (arg == "--uio-memory") o.uio_memory = next();
    else if (arg == "--board") o.board = next();
    else if (arg == "--devmem-regs") o.devmem_regs = number(next());
    else if (arg == "--devmem-memory") {
      const std::string value = next();
      const auto colon = value.find(':');
      if (colon == std::string::npos) throw std::runtime_error("--devmem-memory takes ADDRESS:SIZE");
      o.devmem_phys = number(value.substr(0, colon));
      o.devmem_size = number(value.substr(colon + 1));
    } else if (arg == "--strip-rows") o.strip_rows = static_cast<uint32_t>(number(next()));
    else if (arg == "--timeout") o.timeout = std::stod(next());
    else if (arg == "--sim-memory") o.sim_memory = number(next());   // jopt_driver_sim only
    else throw std::runtime_error("unknown option " + arg + " (--help lists them)");
  }
  if (o.header.empty()) o.header = find_header(fs::path(argv[0]).parent_path());
  if (o.input.empty()) throw std::runtime_error("--input is required (--help lists the options)");
  if (o.board != "zcu104" && o.board != "zynq7030" && o.board != "zynq7020") {
    throw std::runtime_error("--board is zcu104, zynq7030 or zynq7020, not " + o.board);
  }
  if (o.strip_rows % PATCH) {
    throw std::runtime_error("--strip-rows must be a multiple of 32, so that the strips cut whole patches");
  }
  return o;
}

static size_t align_up(size_t bytes) { return (bytes + ALIGN - 1) / ALIGN * ALIGN; }

static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Where the strips go in the reserved DDR: buffers x (input, output).
struct Plan {
  uint32_t strip_rows = 0;
  int buffers = 1;
  size_t in_bytes = 0, out_bytes = 0;   // per buffer, page aligned
  size_t in(int b) const { return b * in_bytes; }
  size_t out(int b) const { return buffers * in_bytes + b * out_bytes; }
};

static Plan plan_strips(uint32_t height, uint32_t width, uint32_t wanted_rows, size_t memory) {
  auto fits = [&](uint32_t rows, int buffers) {
    return buffers * (align_up(rows * width * IN_PIXEL) + align_up(rows * width * OUT_PIXEL)) <= memory;
  };
  Plan p;
  if (wanted_rows == 0 && fits(height, 1)) {
    p.strip_rows = height;   // the whole cube in one start
  } else if (wanted_rows >= height) {
    p.strip_rows = height;
  } else if (wanted_rows) {
    p.strip_rows = wanted_rows;
    p.buffers = 2;
  } else {
    p.buffers = 2;
    for (uint32_t rows = PATCH; rows < height && fits(rows, 2); rows += PATCH) p.strip_rows = rows;
  }
  if (p.strip_rows == 0 || !fits(p.strip_rows, p.buffers)) {
    throw std::runtime_error("the reserved DDR (" + std::to_string(memory) + " bytes) is too small for " +
                             (p.strip_rows ? std::to_string(p.strip_rows) : std::string("32")) +
                             "-row strips of a " + std::to_string(width) + "-pixel-wide cube");
  }
  p.in_bytes = align_up(p.strip_rows * width * IN_PIXEL);
  p.out_bytes = align_up(p.strip_rows * width * OUT_PIXEL);
  return p;
}

static int run(const Options &o) {
  const auto t_total = Clock::now();
  const RegisterMap map = RegisterMap::from_header(o.header);

  uint32_t height = o.height, width = o.width;
  if (!o.capture_config.empty()) read_capture_config(o.capture_config, height, width);
  CubeFile cube(o.input, height, width);
  height = cube.height();
  width = cube.width();

  // The hardware: registers and reserved DDR.
  std::optional<Region> memory;
  std::unique_ptr<Registers> registers;
#ifdef JOPT_SIM
  memory.emplace(Region::heap(0x60000000, o.sim_memory));
  registers = std::make_unique<SimRegisters>(*memory, map);
  std::cout << "simulation: the kernel's C++ on this CPU, " << o.sim_memory << " bytes of \"DDR\"\n";
#else
  if (o.devmem_phys) {
    const uint64_t regs = o.devmem_regs ? *o.devmem_regs : (o.board == "zcu104" ? 0xa0000000 : 0x40000000);
    memory.emplace(Region::devmem(*o.devmem_phys, o.devmem_size));
    registers = std::make_unique<MmioRegisters>(Region::devmem(regs, 0x1000));
    std::cout << "/dev/mem: registers at " << hex(regs) << ", ";
  } else {
    memory.emplace(Region::uio(o.uio_memory, 0));
    registers = std::make_unique<MmioRegisters>(Region::uio(o.uio, 0));
    std::cout << "UIO: registers '" << o.uio << "', ";
  }
  std::cout << "reserved DDR at " << hex(memory->phys()) << ", " << memory->size() / (1 << 20) << " MB\n";
#endif
  Accelerator kernel(*registers, map);

  const Plan plan = plan_strips(height, width, o.strip_rows, memory->size());
  const uint32_t starts = (height + plan.strip_rows - 1) / plan.strip_rows;
  const uint32_t patches = ((height + PATCH - 1) / PATCH) * ((width + PATCH - 1) / PATCH);
  std::cout << height << " x " << width << " pixels, " << patches << " patches, " << kernel.read_ports()
            << " read ports; " << starts << (starts == 1 ? " start" : " starts of " +
               std::to_string(plan.strip_rows) + " rows") << "\n";

  // One strip of the cube in the ARM's memory at a time; the scores of all.
  std::vector<uint64_t> strip((plan.strip_rows * cube.row_bytes() + 7) / 8);
  std::vector<int32_t> scores(size_t(height) * width * 4);
  double load_ms = 0, fpga_ms = 0, read_ms = 0;
  auto rows_of = [&](uint32_t s) { return std::min(plan.strip_rows, height - s * plan.strip_rows); };
  auto fill = [&](uint32_t s) {
    const auto t = Clock::now();
    cube.read_rows(s * plan.strip_rows, rows_of(s), strip.data());
    copy_to(*memory, plan.in(s % plan.buffers), strip.data(), rows_of(s) * cube.row_bytes());
    load_ms += ms_since(t);
  };

  fill(0);
  for (uint32_t s = 0; s < starts; s++) {
    const int b = s % plan.buffers;
    const auto t = Clock::now();
    kernel.start(memory->phys() + plan.in(b), memory->phys() + plan.out(b), rows_of(s), width);
    if (s + 1 < starts) fill(s + 1);   // the other buffer, while the FPGA works
    kernel.wait(o.timeout);
    fpga_ms += ms_since(t);
    const auto t_read = Clock::now();
    copy_from(*memory, plan.out(b), scores.data() + size_t(s) * plan.strip_rows * width * 4,
              size_t(rows_of(s)) * width * OUT_PIXEL);
    read_ms += ms_since(t_read);
  }

  // The class is the kernel's own (lane 3): 0 cloud, 1 land, 2 sea.
  std::vector<uint8_t> labels(size_t(height) * width);
  std::array<size_t, 3> count{};
  size_t invalid = 0;
  for (size_t i = 0; i < labels.size(); i++) {
    const int32_t c = scores[4 * i + 3];
    if (c < 0 || c > 2) {
      invalid++;
      labels[i] = 255;
      continue;
    }
    count[c]++;
    labels[i] = static_cast<uint8_t>(o.hypso_labels ? 2 - c : c);
  }
  const double n = static_cast<double>(labels.size());
  std::printf("classes: cloud %.1f%%, land %.1f%%, sea %.1f%%\n", 100 * count[0] / n, 100 * count[1] / n,
              100 * count[2] / n);
  if (invalid) std::printf("WARNING: %zu pixels have no valid class (written as 255)\n", invalid);

  if (!o.labels.empty()) {
    write_array(o.labels, "|u1", {height, width}, labels.data(), labels.size());
  }
  if (!o.scores.empty()) {
    write_array(o.scores, "<i4", {height, width, 4}, scores.data(), scores.size() * sizeof(int32_t));
  }

  std::printf("cube from file into DDR: %8.1f ms%s\n", load_ms,
              starts > 1 ? " (after the first strip, while the FPGA ran)" : "");
  std::printf("FPGA, start to done:     %8.1f ms, %.1f us per patch%s\n", fpga_ms, 1000 * fpga_ms / patches,
              starts > 1 ? " (with any copy that outlasted the FPGA)" : "");
  std::printf("scores out of DDR:       %8.1f ms\n", read_ms);
  std::printf("wall clock, all of it:   %8.1f ms\n", ms_since(t_total));

  if (!o.reference.empty()) {
    const std::vector<int32_t> want = read_scores_npy(o.reference, height, width);
    size_t wrong = 0, first = 0;
    for (size_t i = 0; i < labels.size(); i++) {
      for (int k = 0; k < 4; k++) {
        if (scores[4 * i + k] != want[4 * i + k]) {
          if (!wrong) first = i;
          wrong++;
          break;
        }
      }
    }
    std::printf("against %s: %s, %zu of %zu pixels differ\n", o.reference.c_str(), wrong ? "FAIL" : "PASS", wrong,
                labels.size());
    if (wrong) {
      std::printf("  first at row %zu, column %zu: kernel %d %d %d %d, reference %d %d %d %d\n", first / width,
                  first % width, scores[4 * first], scores[4 * first + 1], scores[4 * first + 2],
                  scores[4 * first + 3], want[4 * first], want[4 * first + 1], want[4 * first + 2],
                  want[4 * first + 3]);
      return EXIT_MISMATCH;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  try {
    return run(parse(argc, argv));
  } catch (const std::exception &e) {
    std::fprintf(stderr, "jopt_driver: %s\n", e.what());
    return 1;
  }
}
