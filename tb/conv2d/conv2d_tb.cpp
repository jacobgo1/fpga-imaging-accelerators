// conv2d_tb.cpp  --  testbench for the conv2d kernel, not a synthesis source
#include "conv2d.hpp"
#include <cstdio>
#include <random>

static data_t   in    [conv::IC][conv::IH][conv::IW];
static data_t   weight[conv::OC][conv::IC][conv::KH][conv::KW];
static result_t out   [conv::OC][conv::OH][conv::OW];
static result_t ref   [conv::OC][conv::OH][conv::OW];

// Plain, obviously-correct reference. No pragmas, no cleverness.
static void conv2d_ref()
{
    for (int oc = 0; oc < conv::OC; oc++)
        for (int oh = 0; oh < conv::OH; oh++)
            for (int ow = 0; ow < conv::OW; ow++) {
                result_t sum = 0;
                for (int ic = 0; ic < conv::IC; ic++)
                    for (int kh = 0; kh < conv::KH; kh++)
                        for (int kw = 0; kw < conv::KW; kw++)
                            sum += result_t(in[ic][oh * conv::STRIDE + kh]
                                              [ow * conv::STRIDE + kw])
                                 * result_t(weight[oc][ic][kh][kw]);
                ref[oc][oh][ow] = sum;
            }
}

int main()
{
    std::mt19937 rng(1);                              // fixed seed: repeatable
    std::uniform_int_distribution<int> dist(-100, 100);

    for (int ic = 0; ic < conv::IC; ic++)
        for (int h = 0; h < conv::IH; h++)
            for (int w = 0; w < conv::IW; w++)
                in[ic][h][w] = data_t(dist(rng));

    for (int oc = 0; oc < conv::OC; oc++)
        for (int ic = 0; ic < conv::IC; ic++)
            for (int kh = 0; kh < conv::KH; kh++)
                for (int kw = 0; kw < conv::KW; kw++)
                    weight[oc][ic][kh][kw] = data_t(dist(rng));

    conv2d_ref();
    conv2d(in, weight, out);

    int errors = 0;
    for (int oc = 0; oc < conv::OC; oc++)
        for (int oh = 0; oh < conv::OH; oh++)
            for (int ow = 0; ow < conv::OW; ow++)
                if (out[oc][oh][ow] != ref[oc][oh][ow]) {
                    if (errors < 5)
                        printf("mismatch at [%d][%d][%d]: got %lld, expected %lld\n",
                               oc, oh, ow,
                               (long long)out[oc][oh][ow],
                               (long long)ref[oc][oh][ow]);
                    errors++;
                }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", conv::OC * conv::OH * conv::OW);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}