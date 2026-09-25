# Load psu_init + a packed weights/image blob into DDR over JTAG, run
# justoliunet with NO OS on the board at all, and read the result back --
# for a board whose only connection to the build server is the JTAG cable
# (no network, no SD-card Linux to ssh into, so run_justoliunet.py/PYNQ
# don't apply here).
#
# THIS IS A FIRST DRAFT, MORE THAN EVERYTHING ELSE IN THIS DIRECTORY: it was
# written without a real board, Vivado or Vitis install to check `xsct`'s
# exact command syntax against, and justoliunet_regs.tcl's addresses are
# literally placeholders until you fill them in from your own build (see
# justoliunet_regs.tcl.example). Expect to debug this against xsct's actual
# error messages and its own `help <command>` output.
#
# Usage: xsct jtag_run.tcl <data.bin> <psu_init.tcl> <regs.tcl> [hw_server_host:port]
#
# psu_init.tcl: generated from your exported hardware platform (the .xsa
# bitstream --config produces when export_xsa is true). One way to get it:
#   xsct -eval "hsi::open_hw_design system.xsa; hsi::generate_target -dir psu_init {psu_init}"
# then point this script at psu_init/psu_init.tcl. Confirm this against
# your Vitis version's docs if it doesn't match -- the exact hsi command
# for "just give me psu_init" has moved around between releases.

if {$argc < 3} {
    puts stderr "Usage: xsct jtag_run.tcl <data.bin> <psu_init.tcl> <regs.tcl> \[hw_server_host:port\]"
    exit 1
}
set data_bin  [lindex $argv 0]
set psu_init  [lindex $argv 1]
set regs_file [lindex $argv 2]
set hw_server [expr {$argc >= 4 ? [lindex $argv 3] : "localhost:3121"}]

source $regs_file    ;# defines `regs` (dict) and kernel_base / data_base

# Byte offsets of each array inside data.bin -- must match
# pack_data_bin.py's printed table (same fixed geometry, so they will
# unless JNET_H/W/K/BASE_CH/NUM_CLASSES changed on one side and not the
# other).
array set data_offset {
    din 0  w1 1760  b1 1856  w2 1888  b2 3424  w3 3488  b3 8096
    w4 8192  b4 17408  w5 17536  b5 20096  dout 20112
}

connect -url tcp:$hw_server

# 1. Bring up clocks/DDR -- nothing in DDR is valid before this runs (this
#    is normally the FSBL's job on an SD-card boot; there is no FSBL here).
targets -set -filter {name =~ "*PSU*"}
source $psu_init
psu_init
after 1000

# 2. Push the packed weights/image blob into DDR. `dow -data` is xsct's
#    raw binary-into-memory download over the same JTAG link.
targets -set -filter {name =~ "*Cortex-A53 #0*"}
dow -data $data_bin $data_base
puts "Wrote $data_bin to [format 0x%08x $data_base]"

# 3. Point every kernel argument register at its slice of that blob.
proc poke32 {addr value} { mwr $addr $value }
foreach {name} {din w1 b1 w2 b2 w3 b3 w4 b4 w5 b5 dout} {
    set reg_addr  [expr {$kernel_base + [dict get $regs $name]}]
    set data_addr [expr {$data_base + $data_offset($name)}]
    poke32 $reg_addr $data_addr
}

# 4. Start it (ap_start = bit 0) and poll ap_done (bit 1).
poke32 [expr {$kernel_base + [dict get $regs ap_ctrl]}] 1
set done 0
for {set i 0} {$i < 1000} {incr i} {
    set ctrl [mrd -value [expr {$kernel_base + [dict get $regs ap_ctrl]}]]
    if {$ctrl & 0x2} { set done 1; break }
    after 10
}
if {!$done} { error "Timed out waiting for ap_done -- check kernel_base/register offsets in $regs_file" }
puts "Kernel done."

# 5. Read dout back (H*NUM_CLASSES = 16 floats = 64 bytes for the current
#    geometry) and dump it as raw bytes -- decode with unpack_result.py
#    rather than formatting floats in Tcl.
set result_bin "dout.bin"
mrd -bin -file $result_bin [expr {$data_base + $data_offset(dout)}] 16
puts "Wrote $result_bin -- decode with: python3 unpack_result.py $result_bin"

disconnect
