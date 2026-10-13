#!/bin/bash
# Cross-compile jopt_driver for a board's ARM:
#   software/driver/scripts/cross-compile.sh zynq7030     -> software/driver/build-zynq7030/jopt_driver
#   software/driver/scripts/cross-compile.sh zcu104       -> software/driver/build-zcu104/jopt_driver
#   software/driver/scripts/cross-compile.sh zynq7020     -> software/driver/build-zynq7020/jopt_driver (ZedBoard)
# Uses the cross compiler on PATH if there is one (Ubuntu: apt install
# g++-arm-linux-gnueabihf g++-aarch64-linux-gnu), else the Docker image built
# from scripts/Dockerfile.
set -e
board=$1
case "$board" in
    zynq7030|zynq7020) cxx=arm-linux-gnueabihf-g++ ;;
    zcu104)   cxx=aarch64-linux-gnu-g++ ;;
    *) echo "usage: $0 zcu104|zynq7030|zynq7020" >&2; exit 1 ;;
esac
driver=$(cd "$(dirname "$0")/.." && pwd)
build="cmake -S . -B build-$board -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-$board.cmake && cmake --build build-$board -j"

if command -v "$cxx" >/dev/null; then
    (cd "$driver" && eval "$build")
else
    echo "$cxx not on PATH: building in Docker"
    docker image inspect jopt-driver-build >/dev/null 2>&1 || docker build -t jopt-driver-build "$driver/scripts"
    # The whole repository: the driver includes the kernel's header from src/hls/.
    docker run --rm -v "$driver/../..":/repo -w /repo/software/driver --user "$(id -u):$(id -g)" \
        jopt-driver-build bash -c "$build"
fi
file "$driver/build-$board/jopt_driver" 2>/dev/null || ls -l "$driver/build-$board/jopt_driver"
