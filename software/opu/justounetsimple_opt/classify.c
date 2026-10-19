/* classify.c -- runs a justounetsimple_opt kernel on the OPU system's PetaLinux (the
 * lab's ZedBoards, boards/README.md): the notebook's job without PYNQ, Python or numpy.
 * main.py builds it for the board's ARM with the kernel's register offsets
 * (kernel_regs.h, from x<kernel>_hw.h) and puts it in the OPU zip.
 *
 *   classify CUBE.npy SCORES.npy [REFERENCE.npy]
 *
 * CUBE: the raw cube, H x W x 120 uint16 (cube.npy from an image zip). SCORES: what the
 * kernel wrote, H x W x 4 int32: the 3 scores * 2^16, then the class. REFERENCE: the
 * PC's answer (reference_fixed.npy), compared pixel by pixel: PASS or FAIL.
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
    if (uio == 32) fail("no UIO device named kernel: load the design with load.sh first");
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

/* Opens a .npy file and checks its type (e.g. "'<u2'") and number of dimensions; shape
 * gets its size. The file is left at the start of the data. */
static FILE *npy_open(const char *path, const char *descr, int dims, long shape[]) {
    FILE *f = fopen(path, "rb");
    if (!f) fail(path);
    unsigned char start[12];
    if (fread(start, 1, 10, f) != 10 || memcmp(start, "\x93NUMPY", 6) != 0) fail("not a .npy file");
    unsigned long n = start[8] | (unsigned long)start[9] << 8;
    if (start[6] >= 2) {                         /* version 2 and 3: a 4-byte length */
        if (fread(start + 10, 1, 2, f) != 2) fail("a .npy file cut short");
        n |= (unsigned long)start[10] << 16 | (unsigned long)start[11] << 24;
    }
    char *header = malloc(n + 1);
    if (!header || fread(header, 1, n, f) != n) fail("a .npy file cut short");
    header[n] = 0;
    char want[32];
    snprintf(want, sizeof want, "'descr': %s", descr);
    if (!strstr(header, want) || !strstr(header, "'fortran_order': False")) {
        fprintf(stderr, "classify: %s: expected %s in C order, the header is %s\n", path, descr, header);
        exit(1);
    }
    const char *p = strstr(header, "'shape': (");
    if (!p) fail("a .npy header without a shape");
    p += strlen("'shape': (");
    for (int i = 0; i < dims; i++) {
        char *end;
        shape[i] = strtol(p, &end, 10);
        if (end == p) fail("a .npy file with fewer dimensions than expected");
        for (p = end; *p == ',' || *p == ' '; p++) {}
    }
    if (*p != ')') fail("a .npy file with more dimensions than expected");
    free(header);
    return f;
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
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s CUBE.npy SCORES.npy [REFERENCE.npy]\n", argv[0]);
        return 2;
    }
    const double start = now();
    long shape[3], reference_shape[3];
    FILE *in = npy_open(argv[1], "'<u2'", 3, shape);
    const long h = shape[0], w = shape[1];
    if (shape[2] != BANDS) fail("the cube must have 120 bands");
    FILE *reference = NULL;
    if (argc == 4) {
        reference = npy_open(argv[3], "'<i4'", 3, reference_shape);
        if (reference_shape[0] != h || reference_shape[1] != w || reference_shape[2] != LANES)
            fail("the reference is not H x W x 4 for this cube");
    }
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
    int32_t *expected = reference ? malloc(rows * out_row) : NULL;
    if (reference && !expected) fail("no memory for the reference");

    double loading = 0, fpga = 0, saving = 0;
    long patches = 0, wrong = 0, strips = 0;
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

        t = now();
        const int32_t *scores = (const int32_t *)(b.mem + scores_at);
        if (fwrite(scores, out_row, n, out) != (size_t)n) fail("cannot write the scores");
        if (reference) {
            if (fread(expected, out_row, n, reference) != (size_t)n) fail("the reference file is cut short");
            for (long p = 0; p < n * w; p++)
                if (memcmp(scores + LANES * p, expected + LANES * p, LANES * 4) != 0) wrong++;
        }
        saving += now() - t;
        patches += (n + PATCH - 1) / PATCH * ((w + PATCH - 1) / PATCH);
        strips++;
    }
    if (fclose(out) != 0) fail("cannot write the scores");

    printf("%ld x %ld pixels, %ld patches, in %ld start%s of up to %ld rows\n", h, w, patches, strips,
           strips == 1 ? "" : "s", rows);
    printf("cube into memory:   %8.3f s\n", loading);
    printf("FPGA:               %8.1f ms, %.1f us per patch\n", 1e3 * fpga, 1e6 * fpga / patches);
    printf("scores to the file: %8.3f s%s\n", saving, reference ? " (and compared)" : "");
    printf("wall clock:         %8.3f s\n", now() - start);
    if (reference) {
        printf("%s: %ld of %ld pixels differ from %s\n", wrong ? "FAIL" : "PASS", wrong, h * w, argv[3]);
        return wrong ? 1 : 0;
    }
    return 0;
}
