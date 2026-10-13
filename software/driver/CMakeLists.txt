cmake_minimum_required(VERSION 3.13)
project(jopt_driver LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Static, so one binary runs on HYPSO-2's Petalinux and on PYNQ's Ubuntu
# whatever their C libraries' versions.
option(JOPT_STATIC "Link statically (Linux targets)" ON)
# jopt_driver_sim: the kernel's C++ instead of the FPGA, for testing on a PC.
option(JOPT_SIM "Also build jopt_driver_sim" OFF)
set(JOPT_SIM_KERNEL justounetsimple_opt CACHE STRING
    "The kernel jopt_driver_sim runs: justounetsimple_opt or justounetsimple_opt_zynq7030")

set(REPO ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(DRIVER_SOURCES
  src/main.cpp
  src/accelerator.cpp
  src/device.cpp
  src/image_files.cpp
  src/registers.cpp
)

add_executable(jopt_driver ${DRIVER_SOURCES})
target_compile_options(jopt_driver PRIVATE -Wall -Wextra)
# 64-bit file offsets on the 32-bit Zynq-7030 too.
target_compile_definitions(jopt_driver PRIVATE _FILE_OFFSET_BITS=64)
if(JOPT_STATIC AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(jopt_driver PRIVATE -static)
endif()

if(JOPT_SIM)
  file(GLOB KERNEL_SOURCES ${REPO}/src/hls/${JOPT_SIM_KERNEL}/*.cpp)
  add_executable(jopt_driver_sim ${DRIVER_SOURCES} src/sim.cpp ${KERNEL_SOURCES})
  target_include_directories(jopt_driver_sim PRIVATE
    ${REPO}/src/hls/${JOPT_SIM_KERNEL} ${REPO}/src/common)
  target_compile_definitions(jopt_driver_sim PRIVATE JOPT_SIM _FILE_OFFSET_BITS=64)
  if(JOPT_SIM_KERNEL STREQUAL "justounetsimple_opt_zynq7030")
    target_compile_definitions(jopt_driver_sim PRIVATE JOPT_SIM_ZYNQ7030)
  endif()
  target_compile_options(jopt_driver_sim PRIVATE -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label)
endif()
