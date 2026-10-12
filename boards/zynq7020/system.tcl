# boards/zynq7020/system.tcl -- the hardware around the HLS kernel on a Zynq-7020
# (xc7z020clg400-1: the PYNQ-Z2 in the lab, also the Zybo Z7-20). Select it with
# --board zynq7020. A ZedBoard has the same chip in another package: change the
# part in config/project.json to xc7z020clg484-1.
#
# The 7020 is a Zynq-7000 like HYPSO-2's 7030: the same processor (PS7), the
# same four 64-bit HP ports, GP0 and FCLK_CLK0. Only the fabric is smaller (220
# DSPs, 140 BRAM36, 53 K LUTs, no UltraRAM), and that is the kernel's business.
# So the block design is boards/zynq7030/system.tcl's, wire for wire; it takes
# the PS preset (ps7_preset.tcl) and any pins (system.xdc) from this folder.
#
# On a PYNQ board no preset is needed: the PYNQ image has already booted the
# PS, and loading the overlay only sets FCLK_CLK0, from the .hwh.
source [file join [file dirname [file dirname [info script]]] zynq7030 system.tcl]
