# ZCU104 (Zynq UltraScale+): 4 x Cortex-A53, 64-bit ARM.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_CXX_FLAGS_INIT "-mcpu=cortex-a53")
