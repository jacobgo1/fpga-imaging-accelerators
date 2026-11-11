# What every script in scripts/ shares: the repository, a kernel's files, a board's
# settings and the build folder. Sourced, not run.

FPGA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() {
    echo "$(basename "$0"): $*" >&2
    exit 1
}

# Every kernel: one .cpp each in src/models/MODEL/.../, named after it. In byte order (sort
# follows the locale, which puts sp_unet after spectralpointwise1d on some machines).
list_kernels() {
    find "$FPGA_ROOT/src/models" -name '*.cpp' -exec basename {} .cpp \; | LC_ALL=C sort
}

# KERNEL, KERNEL_DIR, SOURCES and TESTBENCH (arrays) of kernel $1: every .cpp in the folder
# holding NAME.cpp is the kernel, every .cpp in tb/NAME/ its testbench.
find_kernel() {
    local name="$1" found
    [[ "$name" =~ ^[A-Za-z0-9_]+$ ]] || die "no kernel '$name': give a kernel's name (scripts/kernels.sh lists them)"
    found="$(find "$FPGA_ROOT/src/models" -name "$name.cpp")"
    [ -n "$found" ] || die "no kernel '$name': expected a file src/models/MODEL/.../$name.cpp (scripts/kernels.sh lists them)"
    [ "$(echo "$found" | wc -l)" -eq 1 ] || die "kernel '$name' is in more than one folder: $found"
    KERNEL="$name"
    KERNEL_DIR="$(dirname "$found")"
    SOURCES=("$KERNEL_DIR"/*.cpp)
    TESTBENCH=()
    [ -d "$FPGA_ROOT/tb/$name" ] && TESTBENCH=("$FPGA_ROOT/tb/$name"/*.cpp)
    [ -f "${TESTBENCH[0]:-}" ] || die "no testbench: add a .cpp with a main() to tb/$name/ that returns nonzero on failure"
}

list_boards() {
    local conf
    for conf in "$FPGA_ROOT"/boards/*/board.conf; do basename "$(dirname "$conf")"; done
}

# BOARD and boards/$1/board.conf's settings: PART, CLOCK_NS, FLOW (pynq or opu), and the
# flow's own.
load_board() {
    local conf="$FPGA_ROOT/boards/$1/board.conf"
    [ -n "$1" ] && [ -f "$conf" ] || die "no board '$1': one of $(list_boards | tr '\n' ' ')"
    BOARD="$1"
    # shellcheck disable=SC1090
    source "$conf"
}

# Where everything for KERNEL on BOARD goes, overwritten by the next build of the same.
build_dir() {
    echo "$FPGA_ROOT/build/$KERNEL/$BOARD"
}

# Cores for Vivado: FPGA_JOBS, else every core this process may use.
job_count() {
    echo "${FPGA_JOBS:-$(nproc 2>/dev/null || echo 4)}"
}

# A value from build.txt (key: value lines, written by hls.sh).
build_info() {
    sed -n "s/^$1: //p" "$(build_dir)/build.txt" 2>/dev/null | tail -n 1
}

# A {braced} Tcl word: taken literally, spaces and all.
tcl_word() {
    case "$1" in *[{}\\]*) die "a path or setting with a brace or backslash cannot go to Tcl: $1" ;; esac
    printf '{%s}' "$1"
}

# A program on PATH, or a message on what to do.
need() {
    command -v "$1" >/dev/null 2>&1 && return
    die "$1 not found on PATH. ${2:-For the AMD tools, source settings64.sh first (fpga_env).}"
}

# The export (export.zip in the build folder) unpacked in DIR: an IP catalog.
unpack_export() {
    local export_zip="$(build_dir)/export.zip"
    [ -f "$export_zip" ] || die "no $export_zip: run scripts/hls.sh $KERNEL $BOARD first, or put someone else's export.zip there"
    rm -rf "$1"
    mkdir -p "$1"
    unzip -q -o "$export_zip" -d "$1"
}
