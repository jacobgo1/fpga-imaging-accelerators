#include "cnn_config.h"
#include <argparse/argparse.hpp>
#include <csignal>
#include <libconfig.h++>
#include <spdlog/sinks/basic_file_sink.h>

// Log signal before exiting program
void logSignal(int sig) {
  spdlog::error("Received termination signal {}", sig);
  exit(sig);
}

// Read command line argument, check arguments, set log config and return the
// rest
bool setProgramConfig(int argc, char *argv[], ProgramConfig &config) {
  std::string log_file = "";
  std::string log_level = "info";

  // Add arguments
  argparse::ArgumentParser program("cnn_accelerator");
  program.add_argument("--input").help(
      "Path to the binned capture in BIP format");
  program.add_argument("--output").help("Path to the output file");
  program.add_argument("--capture-config")
      .help("Path to the capture configuration file")
      .default_value(std::string("capture_config.ini"));
  program.add_argument("--log-level")
      .help("Logging level (trace, debug, info, warn, error, critical)")
      .default_value(log_level);
  program.add_argument("--log-file")
      .help("Path to log file (if not set, logs will be printed to console)")
      .default_value(log_file);

  // Parse arguments and set config
  try {
    program.parse_args(argc, argv);
    config.bip_file = std::filesystem::path(program.get<std::string>("input"));
    config.output_file =
        std::filesystem::path(program.get<std::string>("output"));
    config.capture_config_file =
        std::filesystem::path(program.get<std::string>("capture-config"));
    log_level = program.get<std::string>("log-level");
    spdlog::set_level(spdlog::level::from_str(log_level));
    log_file = program.get<std::string>("log-file");
    if (!log_file.empty()) {
      auto logger = spdlog::basic_logger_mt("cnn-1d", log_file);
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
  if (!std::filesystem::exists(config.capture_config_file)) {
    spdlog::error("Configuration file not found: {}",
                  config.capture_config_file.string());
    return false;
  } else if (!std::filesystem::exists(config.bip_file)) {
    spdlog::error("BIP input file not found: {}", config.bip_file.string());
    return false;
  } else if (!std::filesystem::is_directory(config.output_file.parent_path())) {
    spdlog::error("Parent directory of output file does {} not exist",
                  config.output_file.parent_path().string());
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
