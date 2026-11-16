# Utilization (routed)

Every kernel's resources after place and route, from `reports/impl/utilization.rpt` of its
newest bitstream build, `build/K/RUN/` (RUN in the tables). The reports themselves are in
[`utilization/`](utilization/). Written by `python tools/utilization.py`: do not edit by hand.
BRAM36 counts a RAMB18 as half. These are the real numbers; csynth's estimates are not here.

## ZCU104 (ZU7EV)

The chip has 230 400 LUTs, 460 800 FFs, 312 BRAM36, 96 URAM, 1 728 DSPs.

### Kernel alone

| Kernel | Run | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|---|
| `justounetsimple` | zcu104 | 48 307 (21.0 %) | 43 372 (9.4 %) | 120 (38.5 %) | 2 (2.1 %) | 313 (18.1 %) |
| `justounetsimple_opt` | zcu104 | 37 757 (16.4 %) | 48 930 (10.6 %) | 25.5 (8.2 %) | 16 (16.7 %) | 1 160 (67.1 %) |
| `justounetsimple_opt_tiled` | zcu104 | 42 719 (18.5 %) | 53 983 (11.7 %) | 39.5 (12.7 %) | 0 | 948 (54.9 %) |
| `waveletcnn` | zcu104 | 9 528 (4.1 %) | 12 411 (2.7 %) | 155 (49.7 %) | 58 (60.4 %) | 85 (4.9 %) |
| `waveletcnn_opt` | zcu104 | 28 493 (12.4 %) | 35 221 (7.6 %) | 26 (8.3 %) | 0 | 687 (39.8 %) |
| `sp_unet_small` | zcu104 | 16 617 (7.2 %) | 23 692 (5.1 %) | 32 (10.3 %) | 88 (91.7 %) | 129 (7.5 %) |
| `sp_unet_small_opt` | zcu104 | 76 415 (33.2 %) | 98 967 (21.5 %) | 34 (10.9 %) | 3 (3.1 %) | 1 550 (89.7 %) |
| `sp_unet_small_opt_u8` | zcu104 | 98 303 (42.7 %) | 73 440 (15.9 %) | 42 (13.5 %) | 2 (2.1 %) | 1 387 (80.3 %) |
| `sp_unet_medium` | zcu104 | 18 270 (7.9 %) | 24 413 (5.3 %) | 153 (49.0 %) | 96 (100.0 %) | 129 (7.5 %) |
| `sp_unet_medium_opt` | zcu104 | 93 034 (40.4 %) | 129 403 (28.1 %) | 89 (28.5 %) | 6 (6.2 %) | 1 372 (79.4 %) |
| `sp_unet_medium_opt_u8` | zcu104 | 127 411 (55.3 %) | 95 444 (20.7 %) | 34 (10.9 %) | 3 (3.1 %) | 1 310 (75.8 %) |
| `sp_unet_large` | zcu104 | 20 548 (8.9 %) | 28 967 (6.3 %) | 264.5 (84.8 %) | 96 (100.0 %) | 169 (9.8 %) |
| `sp_unet_large_opt` | zcu104 | 109 789 (47.7 %) | 149 914 (32.5 %) | 85 (27.2 %) | 8 (8.3 %) | 1 273 (73.7 %) |
| `sp_unet_large_opt_u8` | zcu104 | 118 755 (51.5 %) | 103 467 (22.5 %) | 132.5 (42.5 %) | 4 (4.2 %) | 1 213 (70.2 %) |
| `justoliunet` | zcu104 | 25 222 (10.9 %) | 32 926 (7.1 %) | 9.5 (3.0 %) | 0 | 259 (15.0 %) |
| `justoliunet_opt` | zcu104 | 96 403 (41.8 %) | 88 508 (19.2 %) | 10 (3.2 %) | 0 | 1 306 (75.6 %) |
| `justoliunet_bn` | zcu104 | 25 207 (10.9 %) | 32 998 (7.2 %) | 9.5 (3.0 %) | 0 | 259 (15.0 %) |
| `justoliunet_bn_opt` | zcu104 | 98 868 (42.9 %) | 87 688 (19.0 %) | 10 (3.2 %) | 0 | 1 309 (75.8 %) |
| `spectralpointwise1d` | zcu104 | 10 074 (4.4 %) | 16 967 (3.7 %) | 2.5 (0.8 %) | 0 | 129 (7.5 %) |
| `spectralpointwise1d_opt` | zcu104 | 32 634 (14.2 %) | 31 304 (6.8 %) | 18 (5.8 %) | 0 | 1 003 (58.0 %) |

### Whole design: the kernel, interconnects and reset

| Kernel | Run | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|---|
| `justounetsimple` | zcu104 | 49 860 (21.6 %) | 45 354 (9.8 %) | 120 (38.5 %) | 2 (2.1 %) | 313 (18.1 %) |
| `justounetsimple_opt` | zcu104 | 40 264 (17.5 %) | 52 284 (11.3 %) | 25.5 (8.2 %) | 16 (16.7 %) | 1 160 (67.1 %) |
| `justounetsimple_opt_tiled` | zcu104 | 45 219 (19.6 %) | 57 331 (12.4 %) | 39.5 (12.7 %) | 0 | 948 (54.9 %) |
| `waveletcnn` | zcu104 | 11 080 (4.8 %) | 14 388 (3.1 %) | 155 (49.7 %) | 58 (60.4 %) | 85 (4.9 %) |
| `waveletcnn_opt` | zcu104 | 30 998 (13.5 %) | 38 569 (8.4 %) | 26 (8.3 %) | 0 | 687 (39.8 %) |
| `sp_unet_small` | zcu104 | 18 170 (7.9 %) | 25 660 (5.6 %) | 32 (10.3 %) | 88 (91.7 %) | 129 (7.5 %) |
| `sp_unet_small_opt` | zcu104 | 78 921 (34.3 %) | 102 320 (22.2 %) | 34 (10.9 %) | 3 (3.1 %) | 1 550 (89.7 %) |
| `sp_unet_small_opt_u8` | zcu104 | 100 806 (43.8 %) | 76 794 (16.7 %) | 42 (13.5 %) | 2 (2.1 %) | 1 387 (80.3 %) |
| `sp_unet_medium` | zcu104 | 19 823 (8.6 %) | 26 395 (5.7 %) | 153 (49.0 %) | 96 (100.0 %) | 129 (7.5 %) |
| `sp_unet_medium_opt` | zcu104 | 95 545 (41.5 %) | 132 754 (28.8 %) | 89 (28.5 %) | 6 (6.2 %) | 1 372 (79.4 %) |
| `sp_unet_medium_opt_u8` | zcu104 | 129 931 (56.4 %) | 98 792 (21.4 %) | 34 (10.9 %) | 3 (3.1 %) | 1 310 (75.8 %) |
| `sp_unet_large` | zcu104 | 22 098 (9.6 %) | 30 935 (6.7 %) | 264.5 (84.8 %) | 96 (100.0 %) | 169 (9.8 %) |
| `sp_unet_large_opt` | zcu104 | 112 287 (48.7 %) | 153 263 (33.3 %) | 85 (27.2 %) | 8 (8.3 %) | 1 273 (73.7 %) |
| `sp_unet_large_opt_u8` | zcu104 | 121 268 (52.6 %) | 106 815 (23.2 %) | 132.5 (42.5 %) | 4 (4.2 %) | 1 213 (70.2 %) |
| `justoliunet` | zcu104 | 26 775 (11.6 %) | 34 909 (7.6 %) | 9.5 (3.0 %) | 0 | 259 (15.0 %) |
| `justoliunet_opt` | zcu104 | 98 230 (42.6 %) | 90 879 (19.7 %) | 10 (3.2 %) | 0 | 1 306 (75.6 %) |
| `justoliunet_bn` | zcu104 | 26 755 (11.6 %) | 34 966 (7.6 %) | 9.5 (3.0 %) | 0 | 259 (15.0 %) |
| `justoliunet_bn_opt` | zcu104 | 100 700 (43.7 %) | 90 059 (19.5 %) | 10 (3.2 %) | 0 | 1 309 (75.8 %) |
| `spectralpointwise1d` | zcu104 | 11 628 (5.0 %) | 18 935 (4.1 %) | 2.5 (0.8 %) | 0 | 129 (7.5 %) |
| `spectralpointwise1d_opt` | zcu104 | 35 135 (15.2 %) | 34 662 (7.5 %) | 18 (5.8 %) | 0 | 1 003 (58.0 %) |

### Where it goes

Each kernel's processes (inside its DATAFLOW region or golden function, when nearly
all of it is there). Copies of one module are summed (× n); parts under 1 % of the
kernel's LUTs with no DSP or block RAM are summed into "others".

#### `justounetsimple`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| float multipliers (fmul) × 55 | 3 794 | 5 245 | 0 | 0 | 165 |
| float adders (fadd) × 55 | 9 586 | 15 345 | 0 | 0 | 110 |
| justounetsimple_golden_Pipeline_VITIS_LOOP × 5 | 26 382 | 15 818 | 3.5 | 0 | 8 |
| others | 614 | 897 | 0 | 0 | 0 |
| FIFOs and memories | 4 320 | 1 728 | 0 | 0 | 0 |
| glue (the function itself) | 9 | 246 | 0 | 0 | 0 |
| the rest: ports, control, top-level arrays | 1 483 | 2 240 | 101.5 | 1 | 1 |
| **total** | **48 307** | **43 372** | **120** | **2** | **313** |

#### `justounetsimple_opt`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| conv3x3_2mul | 12 356 | 11 754 | 6 | 0 | 864 |
| conv3x3_tiled × 3 | 5 549 | 11 118 | 0 | 0 | 264 |
| patch_buffer | 1 505 | 1 271 | 0 | 16 | 32 |
| read_patch_rows_simple × 4 | 6 915 | 12 587 | 0 | 0 | 0 |
| max_pool × 2 | 1 558 | 1 563 | 0 | 0 | 0 |
| upsample × 2 | 984 | 757 | 0 | 0 | 0 |
| store_scores | 968 | 1 011 | 0 | 0 | 0 |
| others | 72 | 71 | 0 | 0 | 0 |
| FIFOs and memories | 3 630 | 3 479 | 1.5 | 0 | 0 |
| AXI ports (m_axi, s_axi) | 4 052 | 5 250 | 18 | 0 | 0 |
| glue (the function itself) | 103 | 5 | 0 | 0 | 0 |
| **total** | **37 757** | **48 930** | **25.5** | **16** | **1 160** |

#### `justounetsimple_opt_tiled`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| conv3x3_tiled × 4 | 22 809 | 27 913 | 0 | 0 | 912 |
| pixel_buffer | 2 218 | 1 819 | 12 | 0 | 36 |
| read_patch_rows_simple × 4 | 6 931 | 12 579 | 0 | 0 | 0 |
| max_pool × 2 | 1 575 | 1 563 | 0 | 0 | 0 |
| store_scores | 1 028 | 1 011 | 0 | 0 | 0 |
| upsample × 2 | 977 | 757 | 0 | 0 | 0 |
| others | 72 | 71 | 0 | 0 | 0 |
| FIFOs and memories | 2 904 | 2 951 | 9.5 | 0 | 0 |
| AXI ports (m_axi, s_axi) | 4 047 | 5 250 | 18 | 0 | 0 |
| glue (the function itself) | 96 | 5 | 0 | 0 | 0 |
| **total** | **42 719** | **53 983** | **39.5** | **0** | **948** |

#### `waveletcnn`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| float multipliers (fmul) × 15 | 1 022 | 1 470 | 0 | 0 | 45 |
| float adders (fadd) × 13 | 2 329 | 3 691 | 0 | 0 | 26 |
| waveletcnn_golden_float_float | 2 335 | 3 276 | 0 | 0 | 10 |
| waveletcnn_golden_Pipeline_VITIS_LOOP × 3 | 1 288 | 1 602 | 3 | 0 | 3 |
| others | 70 | 36 | 0 | 0 | 0 |
| FIFOs and memories | 577 | 13 | 0 | 57 | 0 |
| glue (the function itself) | 78 | 89 | 0 | 0 | 1 |
| the rest: ports, control, top-level arrays | 1 616 | 2 202 | 128.5 | 1 | 0 |
| **total** | **9 528** | **12 411** | **155** | **58** | **85** |

#### `waveletcnn_opt`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| dwt_proj | 8 123 | 5 175 | 0 | 0 | 352 |
| conv3x3_tiled | 4 128 | 7 524 | 0 | 0 | 265 |
| conv1x1 | 622 | 489 | 0 | 0 | 38 |
| pixel_buffer | 1 295 | 1 070 | 8 | 0 | 32 |
| read_patch_rows_simple × 4 | 6 892 | 12 587 | 0 | 0 | 0 |
| store_scores | 1 036 | 933 | 0 | 0 | 0 |
| others | 72 | 71 | 0 | 0 | 0 |
| FIFOs and memories | 2 152 | 2 076 | 0 | 0 | 0 |
| AXI ports (m_axi, s_axi) | 4 038 | 5 250 | 18 | 0 | 0 |
| glue (the function itself) | 70 | 5 | 0 | 0 | 0 |
| **total** | **28 493** | **35 221** | **26** | **0** | **687** |

#### `sp_unet_small`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| specpw_stem_float_float | 6 453 | 10 757 | 2 | 0 | 81 |
| float multipliers (fmul) × 10 | 695 | 972 | 0 | 0 | 30 |
| float adders (fadd) × 9 | 1 574 | 2 511 | 0 | 0 | 18 |
| specpw_conv_block_float_float × 3 | 2 778 | 2 871 | 9.5 | 5 | 0 |
| specpw_unet_quant_golden_Pipeline_VITIS_LOOP × 5 | 2 117 | 2 695 | 4.5 | 0 | 0 |
| conv2d × 2 | 884 | 951 | 2 | 0 | 0 |
| others | 143 | 139 | 0 | 0 | 0 |
| glue (the function itself) | 8 | 194 | 0 | 0 | 0 |
| the rest: ports, control, top-level arrays | 1 463 | 2 512 | 0.5 | 1 | 0 |
| **total** | **16 617** | **23 692** | **32** | **88** | **129** |

#### `sp_unet_small_opt`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| conv3x3_tiled × 10 | 33 023 | 62 018 | 0 | 0 | 1 275 |
| pointwise_packed | 4 570 | 4 408 | 0 | 0 | 193 |
| conv1x1 × 3 | 2 055 | 2 569 | 0 | 0 | 82 |
| pixel_buffer | 9 786 | 2 874 | 8 | 0 | 0 |
| read_patch_rows_simple × 4 | 6 929 | 12 579 | 0 | 0 | 0 |
| max_pool × 2 | 2 324 | 2 715 | 0 | 0 | 0 |
| store_scores | 1 031 | 1 020 | 0 | 0 | 0 |
| others | 1 939 | 1 448 | 0 | 0 | 0 |
| FIFOs and memories | 8 611 | 3 976 | 8 | 3 | 0 |
| AXI ports (m_axi, s_axi) | 4 037 | 5 250 | 18 | 0 | 0 |
| glue (the function itself) | 2 044 | 5 | 0 | 0 | 0 |
| **total** | **76 415** | **98 967** | **34** | **3** | **1 550** |

#### `sp_unet_small_opt_u8`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| conv3x3_tiled × 10 | 56 057 | 36 992 | 0 | 0 | 923 |
| pointwise_packed | 6 002 | 5 710 | 0 | 0 | 384 |
| conv1x1 × 3 | 2 040 | 2 150 | 0 | 0 | 80 |
| pixel_buffer | 9 919 | 2 993 | 16 | 0 | 0 |
| read_patch_rows_simple × 4 | 6 938 | 12 579 | 0 | 0 | 0 |
| store_scores | 1 019 | 909 | 0 | 0 | 0 |
| max_pool | 991 | 971 | 0 | 0 | 0 |
| others | 2 498 | 1 752 | 0 | 0 | 0 |
| FIFOs and memories | 6 446 | 4 013 | 8 | 2 | 0 |
| AXI ports (m_axi, s_axi) | 4 050 | 5 250 | 18 | 0 | 0 |
| glue (the function itself) | 2 280 | 5 | 0 | 0 | 0 |
| **total** | **98 303** | **73 440** | **42** | **2** | **1 387** |

#### `sp_unet_medium`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| specpw_stem_float_float | 6 058 | 9 515 | 2 | 0 | 61 |
| float multipliers (fmul) × 14 | 977 | 1 366 | 0 | 0 | 42 |
| float adders (fadd) × 13 | 2 267 | 3 627 | 0 | 0 | 26 |
| specpw_conv_block_float_float × 4 | 3 918 | 3 974 | 39.5 | 14 | 0 |
| specpw_unet_quant_golden_Pipeline_VITIS_LOOP × 5 | 2 656 | 3 050 | 15.5 | 0 | 0 |
| others | 238 | 197 | 0 | 0 | 0 |
| FIFOs and memories | 64 | 3 | 0 | 6 | 0 |
| glue (the function itself) | 9 | 196 | 0 | 0 | 0 |
| the rest: ports, control, top-level arrays | 1 434 | 2 436 | 0.5 | 1 | 0 |
| **total** | **18 270** | **24 413** | **153** | **96** | **129** |

#### `sp_unet_medium_opt`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| conv3x3_tiled × 10 | 44 274 | 91 419 | 55 | 0 | 1 221 |
| pointwise_packed | 4 179 | 3 169 | 0 | 0 | 96 |
| conv1x1 × 3 | 1 549 | 1 969 | 0 | 0 | 55 |
| pixel_buffer | 9 752 | 2 811 | 8 | 0 | 0 |
| read_patch_rows_simple × 4 | 6 924 | 12 579 | 0 | 0 | 0 |
| max_pool × 2 | 3 805 | 5 019 | 0 | 0 | 0 |
| store_scores | 1 024 | 1 020 | 0 | 0 | 0 |
| others | 2 194 | 1 990 | 0 | 0 | 0 |
| FIFOs and memories | 13 195 | 4 067 | 8 | 6 | 0 |
| AXI ports (m_axi, s_axi) | 4 064 | 5 250 | 18 | 0 | 0 |
| glue (the function itself) | 2 026 | 5 | 0 | 0 | 0 |
| **total** | **93 034** | **129 403** | **89** | **6** | **1 372** |

#### `sp_unet_medium_opt_u8`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| conv3x3_tiled × 10 | 84 489 | 61 164 | 0 | 0 | 1 161 |
| pointwise_packed | 4 029 | 2 741 | 0 | 0 | 96 |
| conv1x1 × 3 | 1 633 | 1 831 | 0 | 0 | 53 |
| pixel_buffer | 9 740 | 2 811 | 8 | 0 | 0 |
| read_patch_rows_simple × 4 | 6 952 | 12 579 | 0 | 0 | 0 |
| max_pool | 1 566 | 1 739 | 0 | 0 | 0 |
| others | 3 953 | 3 299 | 0 | 0 | 0 |
| FIFOs and memories | 8 681 | 3 920 | 8 | 3 | 0 |
| AXI ports (m_axi, s_axi) | 4 053 | 5 250 | 18 | 0 | 0 |
| glue (the function itself) | 2 253 | 5 | 0 | 0 | 0 |
| **total** | **127 411** | **95 444** | **34** | **3** | **1 310** |

#### `sp_unet_large`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| specpw_stem_float_float | 6 947 | 11 791 | 2 | 0 | 81 |
| float multipliers (fmul) × 18 | 1 211 | 1 750 | 0 | 0 | 54 |
| float adders (fadd) × 17 | 2 962 | 5 031 | 0 | 0 | 34 |
| specpw_conv_block_float_float × 4 | 4 032 | 4 045 | 77 | 18 | 0 |
| specpw_unet_quant_golden_Pipeline_VITIS_LOOP × 5 | 2 844 | 3 375 | 32.5 | 0 | 0 |
| others | 251 | 212 | 0 | 0 | 0 |
| FIFOs and memories | 64 | 3 | 0 | 8 | 0 |
| glue (the function itself) | 7 | 194 | 0 | 0 | 0 |
| the rest: ports, control, top-level arrays | 1 459 | 2 512 | 0.5 | 1 | 0 |
| **total** | **20 548** | **28 967** | **264.5** | **96** | **169** |

#### `sp_unet_large_opt`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| conv3x3_tiled × 10 | 56 862 | 108 566 | 51 | 0 | 1 077 |
| pointwise_packed | 5 345 | 4 083 | 0 | 0 | 128 |
| conv1x1 × 3 | 1 640 | 2 331 | 0 | 0 | 68 |
| pixel_buffer | 9 666 | 2 811 | 8 | 0 | 0 |
| read_patch_rows_simple × 4 | 6 897 | 12 579 | 0 | 0 | 0 |
| max_pool × 2 | 4 791 | 6 555 | 0 | 0 | 0 |
| others | 3 437 | 3 394 | 0 | 0 | 0 |
| FIFOs and memories | 14 913 | 4 235 | 8 | 8 | 0 |
| AXI ports (m_axi, s_axi) | 4 047 | 5 250 | 18 | 0 | 0 |
| glue (the function itself) | 2 166 | 5 | 0 | 0 | 0 |
| **total** | **109 789** | **149 914** | **85** | **8** | **1 273** |

#### `sp_unet_large_opt_u8`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| conv3x3_tiled × 10 | 73 258 | 67 323 | 98.5 | 0 | 1 019 |
| pointwise_packed | 5 126 | 3 439 | 0 | 0 | 128 |
| conv1x1 × 3 | 1 676 | 2 018 | 0 | 0 | 66 |
| pixel_buffer | 9 721 | 2 810 | 8 | 0 | 0 |
| read_patch_rows_simple × 4 | 6 920 | 12 579 | 0 | 0 | 0 |
| max_pool | 1 949 | 2 251 | 0 | 0 | 0 |
| others | 4 198 | 3 747 | 0 | 0 | 0 |
| FIFOs and memories | 9 649 | 3 940 | 8 | 4 | 0 |
| AXI ports (m_axi, s_axi) | 4 047 | 5 250 | 18 | 0 | 0 |
| glue (the function itself) | 2 256 | 5 | 0 | 0 | 0 |
| **total** | **118 755** | **103 467** | **132.5** | **4** | **1 213** |

#### `justoliunet`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| float multipliers (fmul) × 38 | 2 619 | 3 656 | 0 | 0 | 114 |
| float adders (fadd) × 37 | 6 445 | 10 323 | 0 | 0 | 74 |
| justoliunet_golden_float_float | 3 803 | 7 575 | 0 | 0 | 60 |
| justoliunet_golden_Pipeline_VITIS_LOOP × 6 | 8 986 | 6 790 | 3 | 0 | 11 |
| others | 328 | 259 | 0 | 0 | 0 |
| FIFOs and memories | 960 | 1 280 | 0 | 0 | 0 |
| glue (the function itself) | 26 | 197 | 0 | 0 | 0 |
| the rest: ports, control, top-level arrays | 1 887 | 2 814 | 0.5 | 0 | 0 |
| **total** | **25 222** | **32 926** | **9.5** | **0** | **259** |

#### `justoliunet_opt`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| conv1d × 4 | 65 333 | 28 040 | 0 | 0 | 1 298 |
| zscore_pixels | 1 519 | 2 410 | 0 | 0 | 8 |
| pool_to_map × 3 | 6 522 | 17 762 | 0 | 0 | 0 |
| map_windows | 1 194 | 2 144 | 0 | 0 | 0 |
| dense_map | 1 056 | 823 | 0 | 0 | 0 |
| others | 4 445 | 10 394 | 0 | 0 | 0 |
| FIFOs and memories | 10 334 | 22 503 | 0 | 0 | 0 |
| AXI ports (m_axi, s_axi) | 2 391 | 3 168 | 10 | 0 | 0 |
| glue (the function itself) | 3 315 | 3 | 0 | 0 | 0 |
| **total** | **96 403** | **88 508** | **10** | **0** | **1 306** |

#### `justoliunet_bn`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| float multipliers (fmul) × 38 | 2 587 | 3 676 | 0 | 0 | 114 |
| float adders (fadd) × 37 | 6 444 | 10 323 | 0 | 0 | 74 |
| justoliunet_golden_float_float | 3 834 | 7 629 | 0 | 0 | 60 |
| justoliunet_golden_Pipeline_VITIS_LOOP × 6 | 8 977 | 6 788 | 3 | 0 | 11 |
| others | 331 | 259 | 0 | 0 | 0 |
| FIFOs and memories | 960 | 1 280 | 0 | 0 | 0 |
| glue (the function itself) | 22 | 197 | 0 | 0 | 0 |
| the rest: ports, control, top-level arrays | 1 884 | 2 814 | 0.5 | 0 | 0 |
| **total** | **25 207** | **32 998** | **9.5** | **0** | **259** |

#### `justoliunet_bn_opt`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| conv1d × 4 | 66 120 | 27 201 | 0 | 0 | 1 301 |
| zscore_pixels | 1 534 | 2 411 | 0 | 0 | 8 |
| pool_to_map × 3 | 6 544 | 17 762 | 0 | 0 | 0 |
| map_windows | 1 216 | 2 144 | 0 | 0 | 0 |
| dense_map | 1 147 | 846 | 0 | 0 | 0 |
| others | 4 532 | 10 395 | 0 | 0 | 0 |
| FIFOs and memories | 10 297 | 22 497 | 0 | 0 | 0 |
| AXI ports (m_axi, s_axi) | 2 390 | 3 168 | 10 | 0 | 0 |
| glue (the function itself) | 5 190 | 3 | 0 | 0 | 0 |
| **total** | **98 868** | **87 688** | **10** | **0** | **1 309** |

#### `spectralpointwise1d`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| spectralpointwise1d_Pipeline_VITIS_LOOP × 3 | 5 101 | 9 409 | 2 | 0 | 61 |
| float multipliers (fmul) × 14 | 916 | 1 347 | 0 | 0 | 42 |
| float adders (fadd) × 13 | 2 263 | 3 627 | 0 | 0 | 26 |
| others | 38 | 59 | 0 | 0 | 0 |
| FIFOs and memories | 80 | 32 | 0 | 0 | 0 |
| AXI ports (m_axi, s_axi) | 1 595 | 2 144 | 0.5 | 0 | 0 |
| glue (the function itself) | 37 | 280 | 0 | 0 | 0 |
| **total** | **10 074** | **16 967** | **2.5** | **0** | **129** |

#### `spectralpointwise1d_opt`

| Part | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|
| zscore_pointwise × 4 | 18 484 | 13 809 | 0 | 0 | 855 |
| conv1x1 × 8 | 4 905 | 4 864 | 0 | 0 | 148 |
| read_pixels × 4 | 2 086 | 2 126 | 0 | 0 | 0 |
| store_pixels_lanes | 938 | 934 | 0 | 0 | 0 |
| others | 72 | 71 | 0 | 0 | 0 |
| FIFOs and memories | 1 664 | 3 408 | 0 | 0 | 0 |
| AXI ports (m_axi, s_axi) | 3 753 | 5 026 | 18 | 0 | 0 |
| glue (the function itself) | 208 | 5 | 0 | 0 | 0 |
| **total** | **32 634** | **31 304** | **18** | **0** | **1 003** |

## Zynq-7030

The chip has 78 600 LUTs, 157 200 FFs, 265 BRAM36, 400 DSPs.

### Kernel alone

| Kernel | Run | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|---|
| `justounetsimple_opt_zynq7030` | picobob | 51 194 (65.1 %) | 46 403 (29.5 %) | 43 (16.2 %) | 382 (95.5 %) |
| `waveletcnn_opt_zynq7030` | picobob | 26 272 (33.4 %) | 29 475 (18.8 %) | 26.5 (10.0 %) | 334 (83.5 %) |
| `sp_unet_small_opt_u8_zynq7030` | picobob | 48 570 (61.8 %) | 47 957 (30.5 %) | 93.5 (35.3 %) | 374 (93.5 %) |
| `sp_unet_medium_opt_u8_zynq7030` | picobob | 47 133 (60.0 %) | 50 943 (32.4 %) | 123.5 (46.6 %) | 291 (72.8 %) |
| `sp_unet_large_opt_u8_zynq7030` | picobob | 50 432 (64.2 %) | 57 730 (36.7 %) | 116.5 (44.0 %) | 303 (75.8 %) |
| `justoliunet_opt_zynq7030` | picobob | 43 581 (55.4 %) | 45 412 (28.9 %) | 8 (3.0 %) | 378 (94.5 %) |
| `justoliunet_bn_opt_zynq7030` | picobob | 43 774 (55.7 %) | 45 368 (28.9 %) | 8 (3.0 %) | 377 (94.2 %) |
| `spectralpointwise1d_opt_zynq7030` | picobob | 46 877 (59.6 %) | 37 288 (23.7 %) | 20 (7.5 %) | 372 (93.0 %) |

### Whole design: the kernel, interconnects and reset

| Kernel | Run | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|---|
| `justounetsimple_opt_zynq7030` | picobob | 57 921 (73.7 %) | 54 668 (34.8 %) | 44 (16.6 %) | 382 (95.5 %) |
| `waveletcnn_opt_zynq7030` | picobob | 33 010 (42.0 %) | 37 740 (24.0 %) | 27.5 (10.4 %) | 334 (83.5 %) |
| `sp_unet_small_opt_u8_zynq7030` | picobob | 52 894 (67.3 %) | 52 395 (33.3 %) | 94.5 (35.7 %) | 374 (93.5 %) |
| `sp_unet_medium_opt_u8_zynq7030` | picobob | 51 050 (64.9 %) | 55 011 (35.0 %) | 124.5 (47.0 %) | 291 (72.8 %) |
| `sp_unet_large_opt_u8_zynq7030` | picobob | 54 342 (69.1 %) | 61 798 (39.3 %) | 117.5 (44.3 %) | 303 (75.8 %) |
| `justoliunet_opt_zynq7030` | picobob | 47 475 (60.4 %) | 49 482 (31.5 %) | 9 (3.4 %) | 378 (94.5 %) |
| `justoliunet_bn_opt_zynq7030` | picobob | 47 668 (60.6 %) | 49 438 (31.4 %) | 9 (3.4 %) | 377 (94.2 %) |
| `spectralpointwise1d_opt_zynq7030` | picobob | 53 608 (68.2 %) | 45 559 (29.0 %) | 21 (7.9 %) | 372 (93.0 %) |

### Where it goes

Each kernel's processes (inside its DATAFLOW region or golden function, when nearly
all of it is there). Copies of one module are summed (× n); parts under 1 % of the
kernel's LUTs with no DSP or block RAM are summed into "others".

#### `justounetsimple_opt_zynq7030`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv3x3_tiled × 4 | 27 605 | 20 833 | 12 | 378 |
| pixel_buffer | 6 030 | 2 500 | 12 | 4 |
| read_patch_rows_simple × 4 | 6 221 | 11 851 | 0 | 0 |
| max_pool × 2 | 1 850 | 1 556 | 0 | 0 |
| upsample × 2 | 1 090 | 757 | 0 | 0 |
| store_scores | 993 | 1 039 | 0 | 0 |
| others | 0 | 1 | 0 | 0 |
| FIFOs and memories | 2 589 | 2 942 | 13 | 0 |
| AXI ports (m_axi, s_axi) | 3 911 | 4 919 | 6 | 0 |
| glue (the function itself) | 973 | 5 | 0 | 0 |
| **total** | **51 194** | **46 403** | **43** | **382** |

#### `waveletcnn_opt_zynq7030`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| dwt_proj | 3 881 | 2 926 | 0 | 188 |
| conv3x3_tiled | 2 814 | 5 362 | 0 | 109 |
| conv1x1 | 642 | 671 | 0 | 37 |
| read_patch_rows_simple × 4 | 6 255 | 11 851 | 0 | 0 |
| pixel_buffer | 6 122 | 1 780 | 8 | 0 |
| store_scores | 1 030 | 1 005 | 0 | 0 |
| others | 0 | 1 | 0 | 0 |
| FIFOs and memories | 1 236 | 955 | 12.5 | 0 |
| AXI ports (m_axi, s_axi) | 3 896 | 4 919 | 6 | 0 |
| glue (the function itself) | 438 | 5 | 0 | 0 |
| **total** | **26 272** | **29 475** | **26.5** | **334** |

#### `sp_unet_small_opt_u8_zynq7030`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv3x3_tiled_ram × 10 | 26 618 | 22 384 | 71.5 | 276 |
| pointwise_packed | 3 051 | 2 239 | 0 | 64 |
| conv1x1 × 3 | 2 378 | 1 956 | 0 | 26 |
| pixel_buffer | 1 055 | 763 | 4 | 8 |
| read_patch_rows_simple × 2 | 3 161 | 5 931 | 0 | 0 |
| max_pool × 2 | 1 853 | 1 556 | 0 | 0 |
| upsample × 2 | 1 119 | 724 | 0 | 0 |
| store_scores | 975 | 914 | 0 | 0 |
| others | 814 | 437 | 0 | 0 |
| FIFOs and memories | 5 107 | 7 932 | 12 | 0 |
| AXI ports (m_axi, s_axi) | 2 451 | 3 118 | 6 | 0 |
| glue (the function itself) | 139 | 3 | 0 | 0 |
| **total** | **48 570** | **47 957** | **93.5** | **374** |

#### `sp_unet_medium_opt_u8_zynq7030`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv3x3_tiled_ram × 10 | 26 481 | 24 638 | 90 | 252 |
| pointwise_packed | 1 115 | 1 548 | 10 | 20 |
| conv1x1 × 3 | 1 850 | 1 829 | 0 | 11 |
| pixel_buffer | 1 767 | 1 248 | 2.5 | 8 |
| max_pool × 2 | 2 912 | 2 714 | 0 | 0 |
| read_patch_rows_simple | 1 581 | 2 968 | 0 | 0 |
| upsample × 2 | 1 383 | 978 | 0 | 0 |
| store_scores | 893 | 914 | 0 | 0 |
| others | 818 | 437 | 0 | 0 |
| FIFOs and memories | 6 252 | 11 479 | 17 | 0 |
| AXI ports (m_axi, s_axi) | 1 649 | 2 188 | 4 | 0 |
| glue (the function itself) | 503 | 2 | 0 | 0 |
| **total** | **47 133** | **50 943** | **123.5** | **291** |

#### `sp_unet_large_opt_u8_zynq7030`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv3x3_tiled_ram × 10 | 29 062 | 28 262 | 79.5 | 276 |
| pointwise_packed | 1 077 | 1 852 | 8 | 16 |
| conv1x1 × 3 | 1 787 | 1 834 | 3 | 7 |
| pixel_buffer | 937 | 562 | 2 | 4 |
| max_pool × 2 | 3 627 | 3 491 | 0 | 0 |
| read_patch_rows_simple | 1 582 | 2 968 | 0 | 0 |
| upsample × 2 | 1 577 | 1 170 | 0 | 0 |
| store_scores | 974 | 914 | 0 | 0 |
| others | 825 | 437 | 0 | 0 |
| FIFOs and memories | 7 001 | 14 050 | 20 | 0 |
| AXI ports (m_axi, s_axi) | 1 648 | 2 188 | 4 | 0 |
| glue (the function itself) | 442 | 2 | 0 | 0 |
| **total** | **50 432** | **57 730** | **116.5** | **303** |

#### `justoliunet_opt_zynq7030`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv1d × 4 | 27 028 | 21 532 | 0 | 378 |
| zscore_pixels | 2 340 | 2 535 | 0 | 0 |
| dense_positions | 1 679 | 1 307 | 0 | 0 |
| pool1d × 2 | 1 074 | 948 | 0 | 0 |
| store_pixels | 701 | 539 | 0 | 0 |
| read_pixels | 517 | 578 | 0 | 0 |
| others | 2 368 | 5 950 | 0 | 0 |
| FIFOs and memories | 4 568 | 9 931 | 4 | 0 |
| AXI ports (m_axi, s_axi) | 1 582 | 2 090 | 4 | 0 |
| glue (the function itself) | 2 031 | 2 | 0 | 0 |
| **total** | **43 581** | **45 412** | **8** | **378** |

#### `justoliunet_bn_opt_zynq7030`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv1d × 4 | 27 264 | 21 483 | 0 | 377 |
| zscore_pixels | 2 340 | 2 535 | 0 | 0 |
| dense_positions | 1 711 | 1 313 | 0 | 0 |
| pool1d × 2 | 1 076 | 948 | 0 | 0 |
| store_pixels | 701 | 539 | 0 | 0 |
| read_pixels | 516 | 578 | 0 | 0 |
| others | 2 366 | 5 950 | 0 | 0 |
| FIFOs and memories | 4 560 | 9 930 | 4 | 0 |
| AXI ports (m_axi, s_axi) | 1 581 | 2 090 | 4 | 0 |
| glue (the function itself) | 1 900 | 2 | 0 | 0 |
| **total** | **43 774** | **45 368** | **8** | **377** |

#### `spectralpointwise1d_opt_zynq7030`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| zscore_pointwise × 4 | 22 360 | 16 190 | 0 | 372 |
| conv1x1 × 8 | 14 401 | 9 492 | 0 | 0 |
| read_pixels × 4 | 2 107 | 2 178 | 0 | 0 |
| store_pixels_lanes | 975 | 1 081 | 0 | 0 |
| others | 75 | 72 | 0 | 0 |
| FIFOs and memories | 2 771 | 3 632 | 10 | 0 |
| AXI ports (m_axi, s_axi) | 3 550 | 4 638 | 10 | 0 |
| glue (the function itself) | 799 | 5 | 0 | 0 |
| **total** | **46 877** | **37 288** | **20** | **372** |

## Zynq-7020 (ZedBoard, PYNQ-Z2)

The chip has 53 200 LUTs, 106 400 FFs, 140 BRAM36, 220 DSPs.

### Kernel alone

| Kernel | Run | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|---|
| `justounetsimple_opt_zynq7020` | zedboard | 40 588 (76.3 %) | 47 895 (45.0 %) | 26.5 (18.9 %) | 216 (98.2 %) |
| `waveletcnn_opt_zynq7020` | zedboard | 16 658 (31.3 %) | 18 434 (17.3 %) | 18 (12.9 %) | 185 (84.1 %) |
| `sp_unet_small_opt_u8_zynq7020` | zedboard | 39 422 (74.1 %) | 36 188 (34.0 %) | 72 (51.4 %) | 207 (94.1 %) |
| `sp_unet_small_opt_zynq7020` | zedboard | 41 819 (78.6 %) | 45 350 (42.6 %) | 82.5 (58.9 %) | 199 (90.5 %) |
| `justoliunet_opt_zynq7020` | zedboard | 34 331 (64.5 %) | 32 548 (30.6 %) | 8 (5.7 %) | 216 (98.2 %) |
| `justoliunet_bn_opt_zynq7020` | zedboard | 34 594 (65.0 %) | 33 071 (31.1 %) | 8 (5.7 %) | 216 (98.2 %) |
| `spectralpointwise1d_opt_zynq7020` | zedboard | 21 644 (40.7 %) | 15 740 (14.8 %) | 11 (7.9 %) | 196 (89.1 %) |

### Whole design: the kernel, interconnects and reset

| Kernel | Run | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|---|
| `justounetsimple_opt_zynq7020` | zedboard | 45 230 (85.0 %) | 53 218 (50.0 %) | 27.5 (19.6 %) | 216 (98.2 %) |
| `waveletcnn_opt_zynq7020` | zedboard | 21 339 (40.1 %) | 23 757 (22.3 %) | 19 (13.6 %) | 185 (84.1 %) |
| `sp_unet_small_opt_u8_zynq7020` | zedboard | 43 670 (82.1 %) | 41 141 (38.7 %) | 73 (52.1 %) | 207 (94.1 %) |
| `sp_unet_small_opt_zynq7020` | zedboard | 46 039 (86.5 %) | 50 303 (47.3 %) | 83.5 (59.6 %) | 199 (90.5 %) |
| `justoliunet_opt_zynq7020` | zedboard | 38 602 (72.6 %) | 37 503 (35.2 %) | 9 (6.4 %) | 216 (98.2 %) |
| `justoliunet_bn_opt_zynq7020` | zedboard | 38 850 (73.0 %) | 38 026 (35.7 %) | 9 (6.4 %) | 216 (98.2 %) |
| `spectralpointwise1d_opt_zynq7020` | zedboard | 26 328 (49.5 %) | 21 065 (19.8 %) | 12 (8.6 %) | 196 (89.1 %) |

### Where it goes

Each kernel's processes (inside its DATAFLOW region or golden function, when nearly
all of it is there). Copies of one module are summed (× n); parts under 1 % of the
kernel's LUTs with no DSP or block RAM are summed into "others".

#### `justounetsimple_opt_zynq7020`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv3x3_tiled × 4 | 24 499 | 32 424 | 8 | 216 |
| read_patch_rows_simple × 2 | 3 139 | 5 803 | 0 | 0 |
| pixel_buffer | 2 355 | 1 074 | 4 | 0 |
| max_pool × 2 | 1 355 | 1 118 | 0 | 0 |
| store_scores | 1 026 | 918 | 0 | 0 |
| upsample × 2 | 984 | 623 | 0 | 0 |
| others | 1 | 1 | 0 | 0 |
| FIFOs and memories | 4 003 | 2 864 | 10.5 | 0 |
| AXI ports (m_axi, s_axi) | 2 412 | 3 067 | 4 | 0 |
| glue (the function itself) | 990 | 3 | 0 | 0 |
| **total** | **40 588** | **47 895** | **26.5** | **216** |

#### `waveletcnn_opt_zynq7020`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| dwt_proj | 2 232 | 1 673 | 0 | 93 |
| conv3x3_tiled | 2 559 | 4 482 | 0 | 55 |
| conv1x1 | 592 | 523 | 0 | 37 |
| read_patch_rows_simple × 2 | 3 165 | 5 803 | 0 | 0 |
| pixel_buffer | 3 104 | 1 079 | 4 | 0 |
| store_scores | 1 035 | 906 | 0 | 0 |
| others | 0 | 1 | 0 | 0 |
| FIFOs and memories | 1 634 | 897 | 10 | 0 |
| AXI ports (m_axi, s_axi) | 2 421 | 3 067 | 4 | 0 |
| glue (the function itself) | 0 | 3 | 0 | 0 |
| **total** | **16 658** | **18 434** | **18** | **185** |

#### `sp_unet_small_opt_u8_zynq7020`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv3x3_tiled_ram × 10 | 19 601 | 16 015 | 52 | 146 |
| pointwise_packed | 2 293 | 1 710 | 2 | 40 |
| conv1x1 × 3 | 1 910 | 1 561 | 0 | 13 |
| pixel_buffer | 1 884 | 1 063 | 2.5 | 8 |
| read_patch_rows_simple | 1 585 | 2 904 | 0 | 0 |
| max_pool × 2 | 1 280 | 1 118 | 0 | 0 |
| upsample × 2 | 1 037 | 590 | 0 | 0 |
| store_scores | 984 | 827 | 0 | 0 |
| others | 723 | 303 | 0 | 0 |
| FIFOs and memories | 6 446 | 7 907 | 11.5 | 0 |
| AXI ports (m_axi, s_axi) | 1 630 | 2 188 | 4 | 0 |
| glue (the function itself) | 138 | 2 | 0 | 0 |
| **total** | **39 422** | **36 188** | **72** | **207** |

#### `sp_unet_small_opt_zynq7020`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv3x3_tiled_ram × 10 | 20 337 | 19 154 | 46.5 | 164 |
| pointwise_packed | 1 071 | 1 751 | 10 | 20 |
| pixel_buffer | 1 873 | 1 063 | 2.5 | 8 |
| conv1x1 × 3 | 2 075 | 1 938 | 2.5 | 7 |
| max_pool × 2 | 1 861 | 1 985 | 0 | 0 |
| read_patch_rows_simple | 1 586 | 2 904 | 0 | 0 |
| upsample × 2 | 1 282 | 878 | 0 | 0 |
| store_scores | 978 | 865 | 0 | 0 |
| others | 721 | 303 | 0 | 0 |
| FIFOs and memories | 8 393 | 12 319 | 17 | 0 |
| AXI ports (m_axi, s_axi) | 1 647 | 2 188 | 4 | 0 |
| glue (the function itself) | 137 | 2 | 0 | 0 |
| **total** | **41 819** | **45 350** | **82.5** | **199** |

#### `justoliunet_opt_zynq7020`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv1d × 4 | 18 209 | 10 498 | 0 | 216 |
| zscore_pixels | 2 384 | 2 473 | 0 | 0 |
| pool1d × 4 | 1 968 | 1 515 | 0 | 0 |
| dense_positions | 1 592 | 1 128 | 0 | 0 |
| position_windows × 3 | 1 145 | 1 908 | 0 | 0 |
| store_pixels | 661 | 541 | 0 | 0 |
| read_pixels | 533 | 577 | 0 | 0 |
| map_windows | 430 | 1 871 | 0 | 0 |
| others | 74 | 72 | 0 | 0 |
| FIFOs and memories | 4 481 | 9 813 | 4 | 0 |
| AXI ports (m_axi, s_axi) | 1 611 | 2 150 | 4 | 0 |
| glue (the function itself) | 1 460 | 2 | 0 | 0 |
| **total** | **34 331** | **32 548** | **8** | **216** |

#### `justoliunet_bn_opt_zynq7020`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| conv1d × 4 | 18 845 | 11 012 | 0 | 216 |
| zscore_pixels | 2 351 | 2 473 | 0 | 0 |
| pool1d × 4 | 1 917 | 1 515 | 0 | 0 |
| dense_positions | 1 584 | 1 134 | 0 | 0 |
| position_windows × 3 | 1 103 | 1 908 | 0 | 0 |
| store_pixels | 650 | 541 | 0 | 0 |
| read_pixels | 520 | 577 | 0 | 0 |
| map_windows | 420 | 1 871 | 0 | 0 |
| others | 74 | 72 | 0 | 0 |
| FIFOs and memories | 4 488 | 9 816 | 4 | 0 |
| AXI ports (m_axi, s_axi) | 1 611 | 2 150 | 4 | 0 |
| glue (the function itself) | 1 164 | 2 | 0 | 0 |
| **total** | **34 594** | **33 071** | **8** | **216** |

#### `spectralpointwise1d_opt_zynq7020`

| Part | LUTs | FFs | BRAM36 | DSPs |
|---|---|---|---|---|
| zscore_pointwise × 2 | 9 324 | 5 048 | 0 | 196 |
| conv1x1 × 4 | 7 188 | 3 791 | 0 | 0 |
| read_pixels × 2 | 1 065 | 1 092 | 0 | 0 |
| store_pixels_lanes | 812 | 932 | 0 | 0 |
| others | 75 | 72 | 0 | 0 |
| FIFOs and memories | 792 | 1 866 | 5 | 0 |
| AXI ports (m_axi, s_axi) | 2 260 | 2 936 | 6 | 0 |
| glue (the function itself) | 138 | 3 | 0 | 0 |
| **total** | **21 644** | **15 740** | **11** | **196** |

