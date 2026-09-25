# Software

Host utilities, bare-metal applications and embedded Linux components, once a
board and software stack are chosen. Take AXI-Lite register offsets from the
driver headers generated inside the exported HLS IP rather than hardcoding them.

## Talking to a kernel over JTAG (`jtag/`)

The quickest way to put data into a kernel and read results back, with no
embedded software: `xsdb`, the debugger that ships with Vivado, reads and writes
the kernel's AXI-Lite registers through the JTAG cable, via the Zynq PS.

This only works for kernels whose arrays are on the `s_axilite` bundle
(`matmul` is), because then every element is a register. Streaming (`axis`) or
`bram` ports need a DMA in the block design first.

On the machine the board is plugged into, with the Vivado settings sourced
and the board in JTAG boot mode:

```bash
python3 main.py bitstream --kernel matmul        # with board_script = boards/zcu104/system.tcl
xsdb software/jtag/matmul.tcl artifacts/matmul/<run>
```

That programs the bitstream, initializes the PS (clock, resets, PS-to-PL
ports), and runs a self-test against a software reference. No separate Vivado
Hardware Manager step is needed; the script resets the board and programs it
itself.

Live, from the `xsdb` prompt:

```tcl
source software/jtag/matmul.tcl
board_open artifacts/matmul/<run>
set a {{1 2 0 0 0 0 0 0} {0 1 0 0 0 0 0 0} ...}      ;# 8 rows of 8
mm_print [mm_multiply $a [mm_identity]]
mm_print [mm_multiply [mm_random] [mm_random]]
mm_selftest 20
kernel_write a {...64 values...}; kernel_run; kernel_read c -signed   ;# raw access
kernel_status
```

`board.tcl` is kernel-independent: it takes array offsets, widths and depths
from the HLS `*_hw.h` header that `boards/zcu104/system.tcl` copies into
`artifacts/<kernel>/<run>/board/`, next to `psu_init.tcl` and `address.tcl`
(the kernel's base address). If `address.tcl` is missing, pass the base from
the Vivado Address Editor: `board_open <dir> 0xA0000000`.
