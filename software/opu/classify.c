/* classify.c -- runs an optimized kernel on the OPU system's PetaLinux (the lab's ZedBoards
 * and HYPSO-2's PicoBoB): no PYNQ, Python or numpy on the board. Every optimized kernel takes
 * the raw cube and gives the same scores, so one program serves all. main.py builds it for
 * the board's ARM with the kernel's register offsets (kernel_regs.h, from x<kernel>_hw.h)
 * and puts it in the OPU zip.
 *
 *   classify CUBE.bip SCORES.npy HEIGHT WIDTH
 *
 * CUBE: the raw cube, HEIGHT x WIDTH x 120 uint16 with no header, as the OPU saves a capture
 * (hsiN/binned_cube.bip: HEIGHT its frame_count, WIDTH its rows) and as tools/image.py writes
 * a test image (cube.bip, its size in size.txt). SCORES: what the kernel wrote, H x W x 4
 * int32: the 3 scores * 2^OUT_FRAC, then the class (0 cloud, 1 land, 2 sea). It prints the
 * times, and the kernel's own cycle count if it keeps one (REG_CYCLES), summed over the
 * starts. The classes are checked on the PC (tools/opu_results.py), not here.
 *
 * The registers come from UIO (the overlay's "kernel" node, generic-uio), the buffers
 * from the devicetree's reserved memory (CubeDMA's: it must be idle). That memory is
 * mapped uncached, so the kernel reads what the ARM wrote and the ARM reads what the
 * kernel wrote, without cache flushes. A cube too big for it goes in strips of a
 * multiple of 32 rows: the patches are 32 x 32 and independent, so the answers are the
 * same as in one start.
 */
#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef FAKE_BOARD
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "kernel_regs.h"

#define BANDS 120
#define LANES 4
#define PATCH 32
#define TIMEOUT_S 60.0

static void fail(const char *what) {
    fprintf(stderr, "classify: %s\n", what);
    exit(1);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

struct board {
    volatile uint32_t *regs;   /* the kernel's registers */
    uint8_t *mem;              /* the reserved memory, as the ARM sees it */
    uint64_t phys, size;       /* ... and where it is for the kernel */
};

#ifdef FAKE_BOARD
/* The tests' board (tests/test_opu_classify.py): the kernel runs in software. */
void open_board(struct board *b);
void reg_write(struct board *b, unsigned offset, uint32_t value);
uint32_t reg_read(struct board *b, unsigned offset);
#else
static void reg_write(struct board *b, unsigned offset, uint32_t value) { b->regs[offset / 4] = value; }
static uint32_t reg_read(struct board *b, unsigned offset) { return b->regs[offset / 4]; }

/* A short text file's first line, without the newline; 0 if it is not there. */
static int read_line(const char *path, char *text, size_t n) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int got = fgets(text, (int)n, f) != NULL;
    fclose(f);
    if (got) text[strcspn(text, "\n")] = 0;
    return got;
}

static uint64_t big_endian(const unsigned char *p, int bytes) {
    uint64_t v = 0;
    for (int i = 0; i < bytes; i++) v = v << 8 | p[i];
    return v;
}

static void open_board(struct board *b) {
    char path[320], text[64];
    int uio;
    for (uio = 0; uio < 32; uio++) {
        snprintf(path, sizeof path, "/sys/class/uio/uio%d/name", uio);
        if (read_line(path, text, sizeof text) && strcmp(text, "kernel") == 0) break;
    }
    if (uio == 32) fail("no UIO device named kernel: load the design (with its overlay) first");
    snprintf(path, sizeof path, "/sys/class/uio/uio%d/maps/map0/size", uio);
    if (!read_line(path, text, sizeof text)) fail("cannot read the size of the kernel's registers");
    size_t regs_size = strtoul(text, NULL, 0);
    snprintf(path, sizeof path, "/dev/uio%d", uio);
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) fail("cannot open the kernel's UIO device (run as root)");
    void *regs = mmap(NULL, regs_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (regs == MAP_FAILED) fail("cannot map the kernel's registers");
    b->regs = regs;

    /* The first reserved-memory region with a reg: address and size, big-endian cells
     * (one each on a Zynq-7000, two on a 64-bit board). */
    DIR *dir = opendir("/proc/device-tree/reserved-memory");
    if (!dir) fail("no reserved-memory in the devicetree");
    unsigned char cells[16];
    size_t got = 0;
    struct dirent *entry;
    while (got == 0 && (entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        snprintf(path, sizeof path, "/proc/device-tree/reserved-memory/%s/reg", entry->d_name);
        FILE *f = fopen(path, "rb");
        if (f) {
            got = fread(cells, 1, sizeof cells, f);
            fclose(f);
        }
    }
    closedir(dir);
    if (got != 8 && got != 16) fail("no reserved-memory region with a reg in the devicetree");
    b->phys = big_endian(cells, (int)got / 2);
    b->size = big_endian(cells + got / 2, (int)got / 2);
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) fail("cannot open /dev/mem (run as root)");
    void *mem = mmap(NULL, b->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)b->phys);
    if (mem == MAP_FAILED) fail("cannot map the reserved memory");
    b->mem = mem;
}
#endif

/* A positive size from the command line. */
static long size_arg(const char *text, const char *what) {
    char *end;
    const long v = strtol(text, &end, 10);
    if (end == text || *end || v <= 0) {
        fprintf(stderr, "classify: %s must be a positive whole number, not %s\n", what, text);
        exit(2);
    }
    return v;
}

/* The header of an int32 H x W x LANES .npy file: version 1, padded to 64 bytes. */
static void npy_header(FILE *f, long h, long w) {
    char dict[128];
    int n = snprintf(dict, sizeof dict, "{'descr': '<i4', 'fortran_order': False, 'shape': (%ld, %ld, %d), }",
                     h, w, LANES);
    int len = (10 + n + 1 + 63) / 64 * 64 - 10;  /* the dict, spaces, a newline */
    unsigned char start[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, (unsigned char)len, (unsigned char)(len >> 8)};
    fwrite(start, 1, 10, f);
    fputs(dict, f);
    for (int i = n; i < len - 1; i++) fputc(' ', f);
    fputc('\n', f);
}

static void write_address(struct board *b, unsigned offset, int wide, uint64_t address) {
    reg_write(b, offset, (uint32_t)address);
    if (wide) reg_write(b, offset + 4, (uint32_t)(address >> 32));
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s CUBE.bip SCORES.npy HEIGHT WIDTH\n", argv[0]);
        return 2;
    }
    const double start = now();
    const long h = size_arg(argv[3], "HEIGHT"), w = size_arg(argv[4], "WIDTH");

    /* The cube must be exactly H x W x 120 uint16: a wrong size means wrong HEIGHT or WIDTH. */
    FILE *in = fopen(argv[1], "rb");
    if (!in) fail(argv[1]);
    if (fseeko(in, 0, SEEK_END) != 0) fail("cannot find the size of the cube");
    const off_t size = ftello(in), want = (off_t)h * w * BANDS * 2;
    if (size != want) {
        fprintf(stderr, "classify: %s is %lld bytes, but %ld x %ld x %d uint16 is %lld\n", argv[1],
                (long long)size, h, w, BANDS, (long long)want);
        return 1;
    }
    rewind(in);
    FILE *out = fopen(argv[2], "wb");
    if (!out) fail(argv[2]);
    npy_header(out, h, w);

    struct board b;
    open_board(&b);
    if (!(reg_read(&b, REG_AP_CTRL) & 0x4)) fail("the kernel is not idle");

    /* As many rows as fit, in and out, a multiple of 32; the scores 4 KiB aligned after the cube. */
    const size_t in_row = (size_t)w * BANDS * 2, out_row = (size_t)w * LANES * 4;
    long rows = (long)((b.size - 4096) / (in_row + out_row)) / PATCH * PATCH;
    if (rows < PATCH && rows < h) fail("the reserved memory is too small for 32 rows of this cube");
    if (rows > h) rows = h;
    const uint64_t scores_at = ((uint64_t)rows * in_row + 4095) / 4096 * 4096;
    const int dins[] = REG_DINS;

    double loading = 0, fpga = 0, saving = 0;
    long patches = 0, strips = 0;
    unsigned long long cycles = 0;   /* the kernel's own count, every start's */
    for (long y = 0; y < h; y += rows) {
        const long n = h - y < rows ? h - y : rows;
        double t = now();
        if (fread(b.mem, in_row, n, in) != (size_t)n) fail("the cube file is cut short");
        loading += now() - t;

        t = now();
        for (unsigned i = 0; i < sizeof dins / sizeof dins[0]; i++)   /* every input port reads the cube */
            write_address(&b, dins[i], REG_DIN_64, b.phys);
        write_address(&b, REG_DOUT, REG_DOUT_64, b.phys + scores_at);
        reg_write(&b, REG_HEIGHT, (uint32_t)n);
        reg_write(&b, REG_WIDTH, (uint32_t)w);
        reg_write(&b, REG_AP_CTRL, 1);
        while (!(reg_read(&b, REG_AP_CTRL) & 0x2))                   /* done */
            if (now() - t > TIMEOUT_S) fail("the kernel did not finish");
        fpga += now() - t;
#ifdef REG_CYCLES
        cycles += reg_read(&b, REG_CYCLES);
#endif

        t = now();
        if (fwrite(b.mem + scores_at, out_row, n, out) != (size_t)n) fail("cannot write the scores");
        saving += now() - t;
        patches += (n + PATCH - 1) / PATCH * ((w + PATCH - 1) / PATCH);
        strips++;
    }
    if (fclose(out) != 0) fail("cannot write the scores");

    printf("%ld x %ld pixels, %ld patches, in %ld start%s of up to %ld rows\n", h, w, patches, strips,
           strips == 1 ? "" : "s", rows);
    printf("cube into memory:   %8.3f s\n", loading);
    printf("FPGA:               %8.1f ms, %.1f us per patch\n", 1e3 * fpga, 1e6 * fpga / patches);
#ifdef REG_CYCLES
    printf("own count:          %llu cycles\n", cycles);
#endif
    (void)cycles;
    printf("scores to the file: %8.3f s\n", saving);
    printf("wall clock:         %8.3f s\n", now() - start);
    return 0;
}
