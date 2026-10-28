#ifndef SP_UNET_OPT_ZYNQ7020_HLS_HPP
#define SP_UNET_OPT_ZYNQ7020_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// The direct SpecPW-UNet kernels (sp_unet_SIZE_opt_direct, read
// src/hls/sp_unet_small_opt_direct/sp_unet_direct_dataflow.hpp first) sized for
// the Zynq-7020 (the PYNQ-Z2; build with --board zynq7020): sp_unet_small_opt_direct_zynq7020,
// _medium_ and _large_. The same model, the same integer arithmetic, the same
// answers bit for bit, the same raw cube in and scores out; what changes is
// what fits on the chip (boards/README.md): 220 DSP48E1 (these use at most 205),
// 53 k LUTs, 140 BRAM36, no UltraRAM, 64-bit HP ports.
//
//   read_rows_ahead (HP0) ─► pixel_buffer ─► convsp1 ─► convsp2 ─► convsp3 ─►
//   double1 (conv, conv) ─► pool + skip ─┬──► skip FIFO (BRAM) ──────────────────┐
//   double2 (conv, conv) ─► pool + up ───┘                                      │
//   up1 (conv) ─► double3 (conv, conv) ─► up ─► up2 (conv) ─► cat ◄─────────────┘
//   ─► double4 (conv, conv) ─► final 1x1 ─► store (HP1)      every conv with its ReLU
//
// - Every multiplier counts: a conv may take one row of its 3x3 window a step
//   (conv3x3_grouped's KR = 1), so its size goes in steps of 3 DSPs, not 9;
//   convsp1 packs two multiplies per DSP48E1 (16 bits apart, groups of 2) and
//   goes over its output pairs in passes (zynq7020/pointwise_wide.hpp);
//   convsp2/3 and the final layer have time to spare, and take PI inputs a
//   step on 1-4 DSPs (the .cpp's table).
// - LUTs and flip-flops are the scarce part (the first small build needed 66 k
//   LUTs of 53 k): every conv sends ReLU'd int16 activations itself, rounded as
//   each output group finishes (conv3x3_grouped's RELU_SHIFT: PO rounding units,
//   not OUT), so no 32-bit sums cross a FIFO and the pooling only takes maxima;
//   convsp2/3, the final layer and the z-score on a few DSPs (pointwise's PI), not
//   dozens of LUT multipliers; convsp1 rounds a pass's pairs at a time; one read port (the DSPs, not the
//   reads, set the pace: 30 720 cycles a patch of reads against T of 76 000 or
//   more); the FIFOs between the processes 16 deep, as on the ZCU104 (a conv's
//   short wait otherwise freezes the next, 7 % in the cycle model of small),
//   in LUTRAM, cheaper per bit than shift registers.
// - No UltraRAM: the skip connection waits in block RAM.
//
// din0: the raw cube, [height][width][120] uint16. A pixel is 30 words of 4
// bands; words 0-1 (raw 0-7) are dropped, words 2-29 are the kept bands (raw
// 8-117) and raw 118-119, which the z-score zeroes.
// dout: [height][width][4] int32, as the ZCU104 kernels': the 3 scores (score *
// 2^OUT_FRAC), then the class (0 cloud, 1 land, 2 sea).
// cycles: an output register, the kernel's time in clock cycles (count_cycles.hpp).

// The kernels' sizes and formats, used as they are by the .cpp and the
// testbench (the same names as sp_unet_opt.hpp's, other ports and words).
constexpr int H = 32, W = 32;    // patch size
constexpr int RAW_BANDS = 120;   // bands in the cube
constexpr int IN_CH = 110;       // bands the model uses: raw 8-117
constexpr int CLASSES = 3;
constexpr int PORTS = 1;         // read ports
constexpr int PORT_BANDS = 4;    // bands per 64-bit word

constexpr int IN_FRAC = 5;       // the z-scores: x * 2^5 (int8: -3.97 .. +3.97)
constexpr int ACT_FRAC = 10;     // activations (int16: 0 .. 32.0)
constexpr int OUT_FRAC = 16;     // dout = score * 2^16

// One word of the cube: 4 bands of a pixel, 64 bits.
typedef hls::vector<uint16_t, PORT_BANDS> spu_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> spu_out_t;

// A conv layer's size in a kernel's table: PI input x PO output channels x KR
// rows of the 3x3 window a cycle, its multipliers in DSPs or in LUTs.
template<int PI_, int PO_, int KR_, bool DSP_>
struct spu7_layer {
    static constexpr int PI = PI_, PO = PO_, KR = KR_;
    static constexpr bool DSP = DSP_;
};

#endif // SP_UNET_OPT_ZYNQ7020_HLS_HPP
