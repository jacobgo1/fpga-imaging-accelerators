# classify, the program that runs an HLS kernel on the OPU system's PetaLinux, as
# cnn-1d-picobob's CMakeLists.txt builds its cnn_accelerator. scripts/opu.sh builds it,
# with the kernel's HLS driver (from its export.zip) and kernel.h in KERNEL_DRIVER_DIR:
#
#   cmake -S software/opu -B BUILD -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-arm.cmake \
#         -DKERNEL_DRIVER_DIR=build/KERNEL/BOARD/driver
cmake_minimum_required(VERSION 3.14)
project(hls_kernel_app LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add external libraries
include(cmake/spdlog.cmake)
include(cmake/libconfig.cmake)
include(cmake/argparse.cmake)

# The kernel's driver, as HLS generated it (the bare-metal xKERNEL_sinit.c left out)
set(KERNEL_DRIVER_DIR "" CACHE PATH "The kernel's HLS driver and kernel.h")
if(NOT EXISTS "${KERNEL_DRIVER_DIR}/kernel.h")
  message(FATAL_ERROR "No kernel.h in KERNEL_DRIVER_DIR='${KERNEL_DRIVER_DIR}': scripts/opu.sh makes it")
endif()
file(GLOB KERNEL_DRIVER_SRCS ${KERNEL_DRIVER_DIR}/x*.c)
list(FILTER KERNEL_DRIVER_SRCS EXCLUDE REGEX "_sinit\\.c$")
add_library(kernel_driver STATIC ${KERNEL_DRIVER_SRCS})
target_include_directories(kernel_driver PUBLIC ${KERNEL_DRIVER_DIR})
target_compile_definitions(kernel_driver PUBLIC __linux__)
target_compile_options(kernel_driver PRIVATE -Wall)

# The program
add_executable(classify
  src/main.cpp
  src/classify.cpp
  src/kernel_device.cpp
  src/uio_memory_device.cpp
  src/capture_config.cpp
)
target_include_directories(classify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(classify PRIVATE -Wall -Wextra)
target_link_libraries(classify PRIVATE kernel_driver spdlog::spdlog argparse config++)
# The board's C++ library may be older than the compiler's: carry our own.
option(STATIC_LIBSTDCXX "Link the C++ library into classify" ON)
if(STATIC_LIBSTDCXX)
  target_link_options(classify PRIVATE -static-libstdc++ -static-libgcc)
endif()
