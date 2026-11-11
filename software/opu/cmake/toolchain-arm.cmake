SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)

SET(TRIPLE arm-linux-gnueabihf)
SET(CMAKE_C_COMPILER_TARGET ${TRIPLE})
SET(CMAKE_CXX_COMPILER_TARGET ${TRIPLE})

# Use arm-linux-gnueabihf-gcc as compiler
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc -mfpu=neon)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++ -mfpu=neon)
