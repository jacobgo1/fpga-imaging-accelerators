# boards/zedboard/system.tcl -- the hardware around the HLS kernel on a ZedBoard
# (xc7z020clg484-1) running the OPU system's PetaLinux, as the lab's ZedBoards do.
# Select it with --board zedboard.
#
# The same chip as the PYNQ-Z2 (zynq7020) in another package, so the same block
# design: boards/zynq7030/system.tcl's, wire for wire, with the PS preset (if
# any) from this folder. What differs is how it gets onto the board. There is no
# PYNQ: the build packs NAME_opu/ (main.py, package_opu) with the bitstream as
# .bit.bin for fpgautil, its md5sum, and a devicetree overlay that names the
# bitstream and sets FCLK_CLK0, which a bitstream loaded at run time cannot set.
#
# The bitstream replaces the whole PL, the OPU base design (opu_base.tcl: its AXI
# UART, GPIOs and bitstream-hash ROM) included, so load.sh removes that design's
# overlay first. To keep them, the kernel has to go into opu-system's
# ZedBoard_no_comp.tcl instead.
source [file join [file dirname [file dirname [info script]]] zynq7030 system.tcl]
