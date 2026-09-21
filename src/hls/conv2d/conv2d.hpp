#ifndef CONV2D_HPP
#define CONV2D_HPP

#include <cstdint>

// =====================================================================
// 1. Data types
// =====================================================================
using data_t   = std::int16_t;   // input pixels and weights
using result_t = std::int64_t;   // accumulator and output

// =====================================================================
// 2. Layer geometry  --  the only block you normally edit
//
//    These are #defines rather than constexpr on purpose: they are used
//    inside #pragma arguments, and pragmas are parsed from the text the
//    *preprocessor* produces. Macros always work there; constexpr works
//    only sometimes, depending on the Vitis version and the position.
// =====================================================================
#define CONV_IC      3   // input channels
#define CONV_OC      3   // output channels
#define CONV_IH     28   // input height
#define CONV_IW     28   // input width
#define CONV_KH      3   // kernel height
#define CONV_KW      3   // kernel width
#define CONV_STRIDE  1   // stride; no padding ("valid" convolution)
#define CONV_IC_PAR  3   // input channels processed per cycle

// Derived output geometry. Every sub-expression is parenthesised because
// macros are textual substitution -- without the parens, passing
// something like (2+1) as CONV_STRIDE would silently compute nonsense.
#define CONV_OH  (((CONV_IH) - (CONV_KH)) / (CONV_STRIDE) + 1)
#define CONV_OW  (((CONV_IW) - (CONV_KW)) / (CONV_STRIDE) + 1)

// =====================================================================
// 3. Memory banking factors
//
//    These are what make the II=1 schedule possible. They must be at
//    least the kernel size; exactly the kernel size is the cheapest
//    value that still removes every port conflict, so tie them to it.
// =====================================================================
#define CONV_ROW_BANKS  (CONV_KH)
#define CONV_COL_BANKS  (CONV_KW)

// =====================================================================
// 4. Compile-time sanity checks
//    These fire during C simulation, long before you waste 20 minutes
//    on a synthesis run that was doomed from the start.
// =====================================================================
static_assert(CONV_STRIDE >= 1,        "stride must be at least 1");
static_assert(CONV_IH >= CONV_KH,      "kernel is taller than the input");
static_assert(CONV_IW >= CONV_KW,      "kernel is wider than the input");
static_assert(CONV_OH >= 1 && CONV_OW >= 1, "output would be empty");
static_assert(CONV_ROW_BANKS >= CONV_KH, "row banks < kernel height: II=1 is impossible");
static_assert(CONV_COL_BANKS >= CONV_KW, "col banks < kernel width: II=1 is impossible");
static_assert(CONV_IC % CONV_IC_PAR == 0, "CONV_IC_PAR must divide CONV_IC");

// Accumulator width check: worst-case |sum| for the chosen types/sizes.
namespace conv_detail {
constexpr long long kHalfRange = 1LL << (8 * sizeof(data_t) - 1);
constexpr long long kWorstCase =
    (long long)CONV_IC * CONV_KH * CONV_KW * kHalfRange * kHalfRange;
}
static_assert(conv_detail::kWorstCase > 0,"result_t may be too narrow for this configuration");

// =====================================================================
// 5. Plain constexpr mirrors, for testbench / host code that does not
//    want to shout in macros. Not usable inside pragmas.
// =====================================================================
namespace conv {
constexpr int IC = CONV_IC,  OC = CONV_OC;
constexpr int IH = CONV_IH,  IW = CONV_IW;
constexpr int KH = CONV_KH,  KW = CONV_KW;
constexpr int OH = CONV_OH,  OW = CONV_OW;
constexpr int STRIDE = CONV_STRIDE;
}

void conv2d(const data_t in    [CONV_IC][CONV_IH][CONV_IW],
                 const data_t weight[CONV_OC][CONV_IC][CONV_KH][CONV_KW],
                 result_t     out   [CONV_OC][CONV_OH][CONV_OW]);

#endif // CONV2D_HPP