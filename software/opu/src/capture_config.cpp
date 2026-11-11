#include "capture_config.h"
#include <argparse/argparse.hpp>
#include <csignal>
#include <libconfig.h++>
#include <spdlog/sinks/basic_file_sink.h>

// Log signal before exiting program
void logSignal(int sig) {
  spdlog::error("Received termination signal {}", sig);
  exit(sig);
}

static std::optional<std::filesystem::path>
optionalPath(argparse::ArgumentParser &program, const std::string &name) {
  auto value = program.present<std::string>(name);
  if (!value) {
    return std::nullopt;
  }
  return std::filesystem::path(*value);
}

// Read command line argument, check arguments, set log config and return the
// rest
bool setProgramConfig(int argc, char *argv[], ProgramConfig &config) {
  std::string log_file = "";
  std::string log_level = "info";

  // Add arguments
  argparse::ArgumentParser program("classify");
  program.add_description(
      "Runs the HLS kernel on a capture: the raw cube in, the classes out "
      "(sea 0, land 1, cloud 2).");
  program.add_argument("--input").required().help(
      "Path to the capture in BIP format (binned_cube.bip), or a cube .npy "
      "(H x W x 120 uint16)");
  program.add_argument("--output").required().help(
      "Path to the output file: the labels, H x W uint8");
  program.add_argument("--capture-config")
      .help("Path to the capture configuration file (the size of a .bip)");
  program.add_argument("--scores").help(
      "Also write the kernel's scores here, H x W x 4 int32 .npy");
  program.add_argument("--labels").help(
      "Check the classes against these true labels, H x W .npy (0 cloud, 1 "
      "land, 2 sea, others unlabelled)");
  program.add_argument("--log-level")
      .help("Logging level (trace, debug, info, warn, error, critical)")
      .default_value(log_level);
  program.add_argument("--log-file")
      .help("Path to log file (if not set, logs will be printed to console)")
      .default_value(log_file);

  // Parse arguments and set config
  try {
    program.parse_args(argc, argv);
    config.input_file = std::filesystem::path(program.get<std::string>("input"));
    config.output_file =
        std::filesystem::path(program.get<std::string>("output"));
    config.capture_config_file = optionalPath(program, "capture-config");
    config.scores_file = optionalPath(program, "scores");
    config.truth_file = optionalPath(program, "labels");
    log_level = program.get<std::string>("log-level");
    spdlog::set_level(spdlog::level::from_str(log_level));
    log_file = program.get<std::string>("log-file");
    if (!log_file.empty()) {
      auto logger = spdlog::basic_logger_mt("hls-kernel", log_file);
      spdlog::set_default_logger(logger);
    }
    spdlog::flush_on(spdlog::level::err);
    signal(SIGINT, logSignal);
    signal(SIGTERM, logSignal);
    spdlog::debug("Started logger");
  } catch (const std::exception &ex) {
    spdlog::error("Error parsing command line arguments: {}", ex.what());
    return false;
  }

  // Check config
  if (!std::filesystem::exists(config.input_file)) {
    spdlog::error("Input file not found: {}", config.input_file.string());
    return false;
  } else if (config.input_file.extension() != ".npy" &&
             !config.capture_config_file) {
    spdlog::error("A .bip input needs --capture-config for its size");
    return false;
  } else if (config.capture_config_file &&
             !std::filesystem::exists(*config.capture_config_file)) {
    spdlog::error("Configuration file not found: {}",
                  config.capture_config_file->string());
    return false;
  } else if (config.truth_file &&
             !std::filesystem::exists(*config.truth_file)) {
    spdlog::error("Labels file not found: {}", config.truth_file->string());
    return false;
  }
  auto output_dir = std::filesystem::absolute(config.output_file).parent_path();
  if (!std::filesystem::is_directory(output_dir)) {
    spdlog::error("Parent directory of output file does {} not exist",
                  output_dir.string());
    return false;
  }
  return true;
}

CaptureConfig readCaptureConfig(std::filesystem::path filePath) {
  CaptureConfig captureConfig;
  libconfig::Config cfg;

  cfg.readFile(filePath);
  try {
    captureConfig.aoi_y = cfg.lookup("aoi_y");
    captureConfig.frame_count = cfg.lookup("frame_count");
    captureConfig.row_count = cfg.lookup("row_count");
    captureConfig.exposure = cfg.lookup("exposure");
  } catch (const libconfig::SettingNotFoundException &ex) {
    spdlog::error("Missing required configuration parameter: {}", ex.getPath());
    throw;
  } catch (const libconfig::SettingTypeException &ex) {
    if (std::string(ex.getPath()) == "exposure") {
      try {
        int exposure_int = cfg.lookup("exposure");
        captureConfig.exposure = static_cast<float>(exposure_int);
      } catch (const std::exception &ex2) {
        throw std::runtime_error(
            "Invalid type for 'exposure' parameter: expected float or int");
      }
    } else {
      spdlog::error("Invalid type for configuration parameter '{}'",
                    ex.getPath());
      throw;
    }
  }
  return captureConfig;
}
