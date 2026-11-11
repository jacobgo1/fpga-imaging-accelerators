cmake_minimum_required(VERSION 3.13)
project(cnn_accelerator_app LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

# Add external libraries
include(cmake/spdlog.cmake)
include(cmake/libconfig.cmake)
include(cmake/argparse.cmake)

# Create xcnn driver library
set(XCNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/xcnn_driver)
# The kernel's HLS driver, named after the kernel: scripts/opu.sh puts it here
file(GLOB XCNN_SRCS ${XCNN_DIR}/x*.c)
list(FILTER XCNN_SRCS EXCLUDE REGEX "_sinit\\.c$")
add_library(xcnn_driver STATIC ${XCNN_SRCS})
target_include_directories(xcnn_driver PUBLIC ${XCNN_DIR})
target_compile_definitions(xcnn_driver PUBLIC __linux__)
target_compile_options(xcnn_driver PUBLIC -Wall -Wextra)

# Create test executable
add_executable(cnn_accelerator
  src/main.cpp
  src/xcnn_device.cpp
  src/uio_memory_device.cpp
  src/cnn_config.cpp
)
target_include_directories(cnn_accelerator PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${XCNN_DIR}/
)
target_compile_definitions(cnn_accelerator PRIVATE __linux__)
target_compile_options(cnn_accelerator PRIVATE -Wall -Wextra)
target_link_libraries(cnn_accelerator PRIVATE xcnn_driver spdlog::spdlog argparse config++)
