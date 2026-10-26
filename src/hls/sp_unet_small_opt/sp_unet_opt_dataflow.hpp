#pragma once

// The optimized SpecPW-UNet's DATAFLOW chain, shared by sp_unet_small_opt,
// sp_unet_medium_opt and sp_unet_large_opt (the same network in three widths).
// Before including it, a kernel's .cpp includes its model's int8 weights as
// namespace q, and its table of sizes as namespace cfg (each layer's PI x PO,
// and where its multipliers go). sp_unet_opt.hpp has the design.

#include "sp_unet_opt.hpp"
#include "../justounetsimple_opt/justounetsimple_prep.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/shared/pixel_buffer.hpp"
#include "../../optimized/zcu104/pointwise_wide.hpp"
#include "../../optimized/shared/pointwise.hpp"
#include "../../optimized/shared/wino_conv3x3.hpp"
#include "../../optimized/shared/pool_blocks.hpp"
#include "../../optimized/shared/blocks_to_rows.hpp"
#include "../../optimized/shared/concat_blocks.hpp"
#include "../../optimized/shared/store_scores.hpp"
#include "../../optimized/shared/count_cycles.hpp"
#include <type_traits>

// The preprocessing is justounetsimple_opt's (tools/specpw_unet_model.py uses
// its preprocess_fixed): the same integer z-score constants.
namespace prep = justounetsimple_prep;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                               decltype(q::convsp1_weight)>::type>::type,
                           wgt_t>::value,
              "the weights must be int8; did the header's element type change?");
static_assert(prep::IN_FRAC == IN_FRAC && prep::IN_MAX == 127,
              "justounetsimple_prep.hpp is out of date: run python tools/justounetsimple_model.py");

// The channel counts, from the weights (int8: one byte each).
constexpr int S1 = sizeof(q::convsp1_bias), S2 = sizeof(q::convsp2_bias), S3 = sizeof(q::convsp3_bias);
constexpr int C1 = sizeof(q::double1_block_0_bias), C2 = sizeof(q::double2_block_0_bias);
constexpr int C3 = sizeof(q::double3_block_0_bias), C4 = sizeof(q::double4_block_0_bias);
static_assert(sizeof(q::convsp1_weight) == S1 * IN_CH && sizeof(q::final_bias) == CLASSES &&
                  sizeof(q::double4_block_0_weight) == C4 * (C3 + C1) * 9,
              "the weights are not SpecPW-UNet's");

constexpr int WORDS = RAW_BANDS / PORT_BANDS;        // 15 words per pixel in the cube
constexpr int FIRST = prep::FIRST_BAND / PORT_BANDS; // word 0 (raw bands 0-7) is dropped
constexpr int KEPT = WORDS - FIRST;                  // words 1-14: kept bands 0-111
constexpr int CHUNKS = (IN_CH + cfg::P1 - 1) / cfg::P1;   // convsp1's chunks a pixel
static_assert(RAW_BANDS % PORT_BANDS == 0 && cfg::P1 % PORT_BANDS == 0, "a chunk is whole words");
static_assert(prep::FIRST_BAND % PORT_BANDS == 0 && KEPT * PORT_BANDS >= IN_CH &&
                  (KEPT * PORT_BANDS + cfg::P1 - 1) / cfg::P1 == CHUNKS &&
                  sizeof(prep::A) / sizeof(prep::A[0]) == KEPT * PORT_BANDS,
              "the kept bands must be whole words, to the end of each pixel, in convsp1's chunks");

// A layer's fraction bits: its sums carry its input's plus its weights'; the
// bias is shifted up to them, the sums rounded off to the next activations'.
template<int IN_F, int W_F, int B_F, int OUT_F>
struct spu_fracs {
    static constexpr int SUM = IN_F + W_F, BIAS_SHIFT = SUM - B_F, SHIFT = SUM - OUT_F;
    static_assert(BIAS_SHIFT >= 0, "a bias has more fraction bits than its layer's sums");
    static_assert(SHIFT >= 0, "a requantization would have to add bits");
};
#define SPU_FRACS(name, layer, in_frac, out_frac) \
    typedef spu_fracs<in_frac, q::layer##_weight_frac, q::layer##_bias_frac, out_frac> name
SPU_FRACS(F_SP1, convsp1, IN_FRAC, ACT_FRAC);
SPU_FRACS(F_SP2, convsp2, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_SP3, convsp3, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D1A, double1_block_0, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D1B, double1_block_3, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D2A, double2_block_0, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D2B, double2_block_3, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_UP1, up1_1, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D3A, double3_block_0, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D3B, double3_block_3, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_UP2, up2_1, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D4A, double4_block_0, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D4B, double4_block_3, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_FIN, final, ACT_FRAC, OUT_FRAC);
#undef SPU_FRACS

// A Winograd layer's template arguments from its table entry and fractions.
#define SPU_WINO(L, F, layer) cfg::L::PI, cfg::L::PO, F::BIAS_SHIFT, F::SHIFT, \
                              q::layer##_weight, q::layer##_bias, cfg::L::DSP

static void sp_unet_opt_dataflow(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_raw_t> din1,
                                 hls::burst_maxi<spu_raw_t> din2, hls::burst_maxi<spu_raw_t> din3,
                                 hls::burst_maxi<spu_out_t> dout, int height, int width, uint32_t& cycles)
{
#pragma HLS DATAFLOW
    hls::stream<spu_raw_t> s_raw[PORTS];
    hls::stream<vec_t<act8_t, cfg::P1> > s_bands("s_bands");
    hls::stream<vec_t<act_t, S1> > s_sp1("s_sp1");
    hls::stream<vec_t<act_t, S2> > s_sp2("s_sp2");
    hls::stream<vec_t<act_t, 4 * S3> > s_stem("s_stem");          // blocks from here on, unless noted
    hls::stream<vec_t<act_t, 4 * C1> > s_d1a("s_d1a"), s_d1b("s_d1b"), s_skip("s_skip"), s_pool1("s_pool1");
    hls::stream<vec_t<act_t, 4 * C2> > s_d2a("s_d2a"), s_d2b("s_d2b"), s_up1("s_up1");
    hls::stream<vec_t<act_t, C2> > s_pool2("s_pool2");              // pixels
    hls::stream<vec_t<act_t, 4 * C3> > s_d3a("s_d3a"), s_d3b("s_d3b"), s_up2("s_up2");
    hls::stream<vec_t<act_t, C3> > s_dec3("s_dec3");                // pixels
    hls::stream<vec_t<act_t, 4 * (C3 + C1)> > s_cat("s_cat");
    hls::stream<vec_t<act_t, 4 * C4> > s_d4a("s_d4a"), s_d4b("s_d4b");
    hls::stream<vec_t<act_t, C4> > s_dec4("s_dec4");                // pixels
    hls::stream<vec_t<acc_t, CLASSES> > s_scores("s_scores");
    hls::stream<bool> s_finished("s_finished");
// Let the readers run ahead of pixel_buffer, which takes a word from all four at once.
#pragma HLS STREAM variable = s_raw depth = 64
#pragma HLS STREAM variable = s_bands depth = 64
// The skip connection: double1's blocks wait here until up2's come back: up to
// 248 blocks (15.5 block rows of 16) later. Too shallow and the encoder stops while the
// decoder waits for it: a deadlock. 512 blocks fit the UltraRAM's depth anyway.
#pragma HLS STREAM variable = s_skip depth = 512
#pragma HLS BIND_STORAGE variable = s_skip type = fifo impl = uram
// The pooled blocks come in bursts, all of them while the odd block rows go
// by; blocks_to_rows takes a block row at once, then gives pixels for twice as
// long. Room for a block row each, so that the layer before never waits.
#pragma HLS STREAM variable = s_pool1 depth = 8
#pragma HLS STREAM variable = s_d3b depth = 8
#pragma HLS STREAM variable = s_d4b depth = 16
#pragma HLS BIND_STORAGE variable = s_pool1 type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_d3b type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_d4b type = fifo impl = srl
// Pixel rows, so that each wino_up_conv3x3 and the final layer keep going.
#pragma HLS STREAM variable = s_dec3 depth = 32
#pragma HLS STREAM variable = s_dec4 depth = 64
// A patch row of scores, so the final layer keeps going while store starts a burst.
#pragma HLS STREAM variable = s_scores depth = 32
// The other block streams are a block or two wide (up to 4 096 bits), from
// one layer to the next at the same rate: 2 deep, in LUTs, not block RAM.
#pragma HLS BIND_STORAGE variable = s_stem type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_d1a type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_d1b type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_d2a type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_d2b type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_up1 type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_d3a type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_up2 type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_cat type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_d4a type = fifo impl = srl

    // Every process walks the same grid of H x W patches, one after another.
    read_rows_ahead<H, W, WORDS, PORTS, 0>(din0, s_raw[0], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 1>(din1, s_raw[1], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 2>(din2, s_raw[2], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 3>(din3, s_raw[3], height, width);
    // The z-score, then each pixel's P1-band chunks, in 2x2-block order.
    pixel_buffer<H, W, WORDS, FIRST, PORT_BANDS, cfg::P1, PORTS, CHUNKS, prep::FRAC, prep::IN_MAX,
                 prep::A, prep::B, cfg::ZSCORE_DSP, spu_raw_t, true>(s_raw, s_bands, height, width);

    // The spectral stem, 1x1 convs per pixel: convsp1 110 -> S1 on P1 bands a
    // cycle, S1 / 2 x P1 DSPs; convsp2 and convsp3 PO outputs a cycle, in LUTs.
    pointwise_wide<H, W, H * W, IN_CH, S1, cfg::P1, F_SP1::BIAS_SHIFT, F_SP1::SHIFT,
                   q::convsp1_weight, q::convsp1_bias>(s_bands, s_sp1, height, width);
    pointwise<H, W, H * W, S1, S2, cfg::SP2_PO, 1, true, F_SP2::BIAS_SHIFT, F_SP2::SHIFT,
              q::convsp2_weight, q::convsp2_bias, false>(s_sp1, s_sp2, height, width);
    pointwise<H, W, H * W, S2, S3, cfg::SP3_PO, 4, true, F_SP3::BIAS_SHIFT, F_SP3::SHIFT,
              q::convsp3_weight, q::convsp3_bias, false>(s_sp2, s_stem, height, width);

    // The encoder: double1 at 32 x 32, the skip; double2 at 16 x 16.
    wino_conv3x3<H, W, H, W, S3, C1, SPU_WINO(D1A, F_D1A, double1_block_0)>(s_stem, s_d1a, height, width);
    wino_conv3x3<H, W, H, W, C1, C1, SPU_WINO(D1B, F_D1B, double1_block_3)>(s_d1a, s_d1b, height, width);
    pool_skip_blocks<H, W, H, W, C1>(s_d1b, s_skip, s_pool1, height, width);
    wino_conv3x3<H, W, H / 2, W / 2, C1, C2, SPU_WINO(D2A, F_D2A, double2_block_0)>(s_pool1, s_d2a, height, width);
    wino_conv3x3<H, W, H / 2, W / 2, C2, C2, SPU_WINO(D2B, F_D2B, double2_block_3)>(s_d2a, s_d2b, height, width);
    pool_blocks<H, W, H / 2, W / 2, C2>(s_d2b, s_pool2, height, width);          // -> 8 x 8

    // The decoder: up1 8 x 8 -> 16 x 16, double3, up2 -> 32 x 32.
    wino_up_conv3x3<H, W, H / 4, W / 4, C2, C2, SPU_WINO(UP1, F_UP1, up1_1)>(s_pool2, s_up1, height, width);
    wino_conv3x3<H, W, H / 2, W / 2, C2, C3, SPU_WINO(D3A, F_D3A, double3_block_0)>(s_up1, s_d3a, height, width);
    wino_conv3x3<H, W, H / 2, W / 2, C3, C3, SPU_WINO(D3B, F_D3B, double3_block_3)>(s_d3a, s_d3b, height, width);
    blocks_to_rows<H, W, H / 2, W / 2, C3, act_t>(s_d3b, s_dec3, height, width);
    wino_up_conv3x3<H, W, H / 2, W / 2, C3, C3, SPU_WINO(UP2, F_UP2, up2_1)>(s_dec3, s_up2, height, width);

    // The skip connection, then double4 and the final 1x1 conv: the scores.
    concat_blocks<H, W, H, W, C3, C1>(s_up2, s_skip, s_cat, height, width);
    wino_conv3x3<H, W, H, W, C3 + C1, C4, SPU_WINO(D4A, F_D4A, double4_block_0)>(s_cat, s_d4a, height, width);
    wino_conv3x3<H, W, H, W, C4, C4, SPU_WINO(D4B, F_D4B, double4_block_3)>(s_d4a, s_d4b, height, width);
    blocks_to_rows<H, W, H, W, C4, act_t>(s_d4b, s_dec4, height, width);
    pointwise<H, W, H * W, C4, CLASSES, cfg::FIN_PO, 1, false, F_FIN::BIAS_SHIFT, F_FIN::SHIFT,
              q::final_weight, q::final_bias, false>(s_dec4, s_scores, height, width);
    store_scores<H, W, F_FIN::SHIFT>(s_scores, dout, height, width, s_finished);
    // Last: C simulation runs the processes in this order (count_cycles.hpp).
    count_cycles(s_finished, cycles);
}

#undef SPU_WINO
