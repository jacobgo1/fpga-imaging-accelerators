#!/bin/bash
# make_kernel_header.sh DRIVER_DIR CLOCK_MHZ > DRIVER_DIR/kernel.h
#
# kernel.h for cnn_accelerator (src/xcnn_device.cpp): the HLS-generated driver in
# DRIVER_DIR (xKERNEL.h, xKERNEL.c, xKERNEL_linux.c from the export's
# drivers/KERNEL_v1_0/src/) under names that do not depend on the kernel, so the same
# program serves every kernel. CLOCK_MHZ: the clock the kernel was built for, for the log.
# A kernel it can run has the inputs din0, din1, ... (each reads the same cube), dout,
# height and width; cycles if it keeps its own count.
set -e
fail() {
    echo "make_kernel_header.sh: $*" >&2
    exit 1
}
[ $# -eq 2 ] || fail "usage: make_kernel_header.sh DRIVER_DIR CLOCK_MHZ > kernel.h"
header=""
for file in "$1"/x*.h; do
    case "$file" in *_hw.h) ;; *) [ -f "$file" ] && header="$file" ;; esac
done
[ -n "$header" ] || fail "no HLS driver header (x*.h) in $1"
# The driver's prefix: XKernel_name in XKernel_name_Start(...).
prefix="$(grep -oE '\bX[A-Za-z0-9_]+_Start\(' "$header" | head -n 1 | sed 's/_Start($//')"
[ -n "$prefix" ] || fail "$header: no ..._Start function"
has() { grep -qE "\b${prefix}_$1\(" "$header"; }
for function in Set_din0 Set_dout Set_height Set_width; do
    has "$function" || fail "$header: no ${prefix}_$function: cnn_accelerator runs kernels with the inputs din0, ..., dout, height and width (an optimized kernel)"
done
dins="$(grep -oE "\b${prefix}_Set_din[0-9]+\(" "$header" | sed "s/^${prefix}_Set_//; s/($//" | sort -u | sort -t n -k 2 -n)"

echo "// $prefix's HLS driver under the names src/xcnn_device.cpp uses. Made by"
echo "// software/opu/scripts/make_kernel_header.sh from $(basename "$header")."
echo "#pragma once"
echo "#include <cstdint>"
echo "#include \"$(basename "$header")\""
echo
echo "#define KERNEL_NAME \"$(basename "$header" .h | sed 's/^x//')\""
echo "#define KERNEL_CLOCK_MHZ $2"
echo "using KernelInstance = $prefix;"
echo
echo "inline int kernelInitialize(KernelInstance *k, const char *uio_name) { return ${prefix}_Initialize(k, uio_name); }"
echo "inline int kernelRelease(KernelInstance *k) { return ${prefix}_Release(k); }"
echo "// Every input port reads the same cube."
echo "inline void kernelSetInputs(KernelInstance *k, uint64_t cube) {"
for din in $dins; do echo "    ${prefix}_Set_$din(k, cube);"; done
echo "}"
echo "inline void kernelSetOutput(KernelInstance *k, uint64_t scores) { ${prefix}_Set_dout(k, scores); }"
echo "inline void kernelSetSize(KernelInstance *k, uint32_t height, uint32_t width) {"
echo "    ${prefix}_Set_height(k, height);"
echo "    ${prefix}_Set_width(k, width);"
echo "}"
echo "inline void kernelStart(KernelInstance *k) { ${prefix}_Start(k); }"
echo "inline bool kernelIsDone(KernelInstance *k) { return ${prefix}_IsDone(k) != 0; }"
echo "inline bool kernelIsIdle(KernelInstance *k) { return ${prefix}_IsIdle(k) != 0; }"
echo "inline bool kernelIsReady(KernelInstance *k) { return ${prefix}_IsReady(k) != 0; }"
if has Get_cycles; then
    echo "// The kernel's own count of its cycles, of the last start."
    echo "inline bool kernelHasCycles() { return true; }"
    echo "inline uint64_t kernelCycles(KernelInstance *k) { return ${prefix}_Get_cycles(k); }"
else
    echo "inline bool kernelHasCycles() { return false; }"
    echo "inline uint64_t kernelCycles(KernelInstance *) { return 0; }"
fi
