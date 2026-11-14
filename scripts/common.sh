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

# Every build of KERNEL on BOARD is a run of its own, build/KERNEL/BOARD/DATE_TIME/, kept so
# builds can be compared; build/KERNEL/BOARD/latest links to the last one written to. One
# board's runs never move another board's latest.
board_dir() {
    echo "$FPGA_ROOT/build/$KERNEL/$BOARD"
}

# BUILD_DIR: a new run, made the latest. hls.sh starts every build with one.
new_build() {
    local base="$(board_dir)" name n=1
    name="$(date '+%Y-%m-%d_%H-%M-%S')"
    while [ -e "$base/$name" ]; do n=$((n + 1)); name="$(date '+%Y-%m-%d_%H-%M-%S')-$n"; done
    mkdir -p "$base/$name"
    BUILD_DIR="$base/$name"
    point_latest
}

# BUILD_DIR: the newest run with an export.zip, for the steps after HLS (pynq.sh, opu.sh), made
# the latest. Someone else's export.zip, put straight in build/KERNEL/BOARD/, starts a run.
last_build() {
    local base="$(board_dir)" run
    if [ -f "$base/export.zip" ]; then
        new_build
        mv "$base/export.zip" "$BUILD_DIR/"
        echo "export: someone else's" > "$BUILD_DIR/build.txt"
        return
    fi
    BUILD_DIR=""
    # Run names sort by date; latest is a link, not a run.
    for run in "$base"/[0-9][0-9][0-9][0-9]-*/; do
        [ -f "$run/export.zip" ] && BUILD_DIR="${run%/}"
    done
    [ -n "$BUILD_DIR" ] || die "no export.zip in $base/*/: run scripts/hls.sh $KERNEL $BOARD first, or put someone else's export.zip in $base/"
    point_latest
}

# build/KERNEL/BOARD/latest -> BUILD_DIR, relative so the build folder can move.
point_latest() {
    local base="$(board_dir)"
    rm -rf "$base/latest"
    ln -s "$(basename "$BUILD_DIR")" "$base/latest" 2>/dev/null \
        || echo "could not link $base/latest; the build is in $BUILD_DIR" >&2
}

build_dir() {
    echo "$BUILD_DIR"
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
    [ -f "$export_zip" ] || die "no $export_zip: run scripts/hls.sh $KERNEL $BOARD first, or put someone else's export.zip in $(board_dir)/"
    rm -rf "$1"
    mkdir -p "$1"
    unzip -q -o "$export_zip" -d "$1"
}
