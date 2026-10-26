#ifndef SP_UNET_OPT_HLS_HPP
#define SP_UNET_OPT_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// The optimized SpecPW-UNet (sp_unet_small_opt, sp_unet_medium_opt,
// sp_unet_large_opt: the same quantized networks as the golden sp_unet_SIZE)
// on a whole raw capture: one start classifies every pixel. The ARM only
// points the kernel at the cube in DDR; the kernel drops bands, z-scores, cuts
// the image into 32 x 32 patches (the last row and column repeated to fill
// the edge patches) and writes each patch's scores back in place. A DATAFLOW
// chain, every process running at once on its own part of the network:
//
//   read_rows_ahead x 4 ─► pixel_buffer ─► convsp1 ─► convsp2 ─► convsp3 ─┐  the stem, 1x1, per pixel
//      HP0-3              z-score, 4-row   packed     pointwise  (+ 2x2   │
//                         strips, blocks   2 / DSP               blocks)  │
//   ┌─────────────────────────────────────────────────────────────────────┘
//   └► double1 (2 x wino) ─► pool+skip ─┬─► double2 (2 x wino) ─► pool ─► up1 (wino up) ─┐
//                                       │ skip FIFO (UltraRAM)                           │
//   ┌───────────────────────────────────┼────────────────────────────────────────────────┘
//   └► double3 (2 x wino) ─► rows ─► up2 (wino up) ─► cat ─► double4 (2 x wino) ─► rows ─► final 1x1 ─► store (HPC0)
//
// - The 3x3 convs are Winograd F(2x2, 3x3) in integers (wino_conv3x3.hpp):
//   16 multiplies per 2x2 output block and channel pair instead of 36, bit
//   for bit the same sums. The maps between them travel in 2x2-block order,
//   so the max-pools are a block each and the Winograd outputs need no
//   reordering.
// - Each UpBlock (nearest x2, then a 3x3 conv) is computed from the low-res
//   map: the upsampled tile has two equal rows and columns, 9 multiplies per
//   output block instead of 36 (wino_up_conv3x3).
// - The stem is 1x1 convs, so it needs no patch buffer: the four readers' rows
//   are put in order in a strip ping-pong buffer of 4 rows (pixel_buffer), and
//   its first layer takes the int8 bands with two multiplies per DSP.
// - The skip connection waits in a FIFO in UltraRAM while the encoder's
//   output goes down to 8 x 8 and back up (up to 248 blocks ahead).
// - Every process is sized for the same time per patch, T (in the kernels'
//   .cpp): PI input x PO output channels a cycle per layer, whatever makes its
//   own time T. optimizations_implemented/sp_unet.md has the table.
//
// Integer fixed point throughout, bit for bit classify_fixed in
// tools/specpw_unet_model.py (and src/golden/specpw_unet_fixed_golden.hpp):
// int8 z-scores with 5 fraction bits (justounetsimple_opt's preprocessing,
// justounetsimple_prep.hpp), int8 weights, int32 sums, int16 activations with
// 10 fraction bits, ReLU'd, rounded half up and saturated, the scores with 16.
//
// din0 .. din3: the same raw cube, [height][width][120] uint16 -- the four ports
// read different rows of it. A pixel is 15 words of 8 bands; word 0 (raw 0-7)
// is dropped, words 1-14 are the kept bands (raw 8-117) and raw 118-119, which
// the z-score zeroes.
// dout: [height][width][4] int32: the 3 scores (score * 2^OUT_FRAC), then the
// class (0 cloud, 1 land, 2 sea; the first highest score). One 128-bit word a
// pixel.
// Any height and width from 1 to 65535. A strip of a larger image works the
// same way: point din at its first row and dout at that row's scores.
// cycles: an output register, the kernel's time in clock cycles, from its
// start until the last score is in memory (count_cycles.hpp). 0 in C
// simulation, where the processes run one after another.

// The kernels' sizes and formats, used as they are by the .cpp and the testbench.
constexpr int H = 32, W = 32;    // patch size
constexpr int RAW_BANDS = 120;   // bands in the cube
constexpr int IN_CH = 110;       // bands the model uses: raw 8-117
constexpr int CLASSES = 3;
constexpr int PORTS = 4;         // read ports, each a quarter of the rows
constexpr int PORT_BANDS = 8;    // bands per 128-bit word

constexpr int IN_FRAC = 5;       // the z-scores: x * 2^5 (int8: -3.97 .. +3.97)
constexpr int ACT_FRAC = 10;     // activations (int16: 0 .. 32.0)
constexpr int OUT_FRAC = 16;     // dout = score * 2^16

// One word of the cube: 8 bands of a pixel, 128 bits.
typedef hls::vector<uint16_t, PORT_BANDS> spu_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> spu_out_t;

// A Winograd layer's size in a kernel's table: PI input x PO output channels
// a cycle, its multipliers in DSPs or in LUTs.
template<int PI_, int PO_, bool DSP_>
struct spu_layer {
    static constexpr int PI = PI_, PO = PO_;
    static constexpr bool DSP = DSP_;
};

#endif // SP_UNET_OPT_HLS_HPP
