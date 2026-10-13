#include "registers.hpp"

#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

RegisterMap RegisterMap::from_header(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot read the register header " + path +
                             " (xjustounetsimple_opt_hw.h from the HLS build)");
  }
  std::stringstream text;
  text << file.rdbuf();
  return from_text(text.str());
}

RegisterMap RegisterMap::from_text(const std::string &text) {
  // Lines like  #define XJUSTOUNETSIMPLE_OPT_CONTROL_ADDR_DIN0_DATA 0x10
  //       and   #define XJUSTOUNETSIMPLE_OPT_CONTROL_BITS_DIN0_DATA 64
  const std::regex addr_line(R"(#define\s+\w+_CONTROL_ADDR_(\w+)\s+(0x[0-9a-fA-F]+))");
  const std::regex bits_line(R"(#define\s+\w+_CONTROL_BITS_(\w+)\s+(\d+))");
  std::map<std::string, uint32_t> addr;
  std::map<std::string, int> bits;
  std::istringstream lines(text);
  std::string line;
  std::smatch m;
  while (std::getline(lines, line)) {
    if (std::regex_search(line, m, addr_line)) {
      addr[m[1]] = static_cast<uint32_t>(std::stoul(m[2], nullptr, 16));
    } else if (std::regex_search(line, m, bits_line)) {
      bits[m[1]] = std::stoi(m[2]);
    }
  }

  RegisterMap map;
  auto take = [&](const std::string &name) {
    auto it = addr.find(name);
    if (it == addr.end()) {
      throw std::runtime_error("the register header has no " + name +
                               ": not a justounetsimple_opt header?");
    }
    Register r{it->second, bits.count(name) ? bits[name] : 32};
    addr.erase(it);
    return r;
  };
  map.ap_ctrl = take("AP_CTRL").offset;
  for (int i = 0; addr.count("DIN" + std::to_string(i) + "_DATA"); i++) {
    map.din.push_back(take("DIN" + std::to_string(i) + "_DATA"));
  }
  if (map.din.empty()) {
    throw std::runtime_error("the register header has no DIN0_DATA: not a justounetsimple_opt header?");
  }
  map.dout = take("DOUT_DATA");
  map.height = take("HEIGHT_DATA");
  map.width = take("WIDTH_DATA");

  // An argument the driver does not know would silently stay 0: refuse instead.
  for (const auto &[name, offset] : addr) {
    if (name.size() > 5 && name.compare(name.size() - 5, 5, "_DATA") == 0) {
      throw std::runtime_error("the kernel has an argument this driver does not set: " + name);
    }
  }
  return map;
}
