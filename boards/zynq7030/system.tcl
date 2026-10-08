# boards/zynq7030/system.tcl -- the hardware around the HLS kernel on a Zynq-7030
# (the FPGA on HYPSO-2), wire by wire. Select it with --board zynq7030.
#
# scripts/vivado.tcl sources this after it has created an empty project for the
# part in config/project.json and put the exported HLS IP in its catalog. Like
# boards/zcu104/system.tcl, it works for any kernel: it wires whatever control
# and m_axi ports the kernel has.
#
#   PS ("ps": 2 x Cortex-A9, DDR controller)
#     FCLK_CLK0     ───────────────────────────┬──► every clock pin below
#     FCLK_RESET0_N ──► rst (proc_sys_reset) ──┴──► every reset pin below
#     M_AXI_GP0  ──► ctrl (SmartConnect)  ──► kernel/s_axi_control   registers
#     S_AXI_HP0  ◄── data0 (SmartConnect) ◄── kernel/m_axi_gmem0     DDR access
#     S_AXI_HP1  ◄── data1 (SmartConnect) ◄── kernel/m_axi_gmem1     (one HP port
#     ...                                    ...                    per m_axi port)
#     IRQ_F2P    ◄── kernel/interrupt
#
# Differences from the ZCU104 that matter to a kernel:
#   - HP ports are 64 bits wide (128 on the ZCU104): half the DDR bandwidth per
#     port at the same clock. A 128-bit m_axi port still works; its SmartConnect
#     splits each beat in two.
#   - The device is much smaller: about 400 DSPs (1,728 on the ZU7EV), 265 BRAM36
#     and no UltraRAM. A kernel sized for the ZCU104 must be sized down for it.
#
# The PS: there are no Vivado board files for HYPSO-2, so its DDR, MIO and clock
# settings cannot come from a board preset. If boards/zynq7030/ps7_preset.tcl
# exists, it is sourced here with the PS block in $ps: put HYPSO-2's own PS
# configuration in it as set_property -dict [list CONFIG.PCW_... ] $ps lines
# (from the flight design's block design Tcl, write_bd_tcl). Without it the PS
# keeps Vivado's defaults. That does not change the bitstream -- the boot loader
# (FSBL) configures the PS, not the .bit -- but the running system must give the
# PL what this design assumes: FCLK_CLK0 at the HLS clock, and the HP ports used
# here enabled at 64 bits. Check that against HYPSO-2's boot configuration.

# ---------------------------------------------------------------- Blocks
create_bd_design system

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps]
set preset [file join $cfg(root) boards zynq7030 ps7_preset.tcl]
if {[file exists $preset]} {
    puts "INFO: PS configuration from $preset"
    source $preset
} else {
    puts "WARNING: no boards/zynq7030/ps7_preset.tcl: the PS keeps Vivado's default DDR/MIO settings.\
          The bitstream does not depend on them, but HYPSO-2's boot configuration must match\
          the clock and HP ports below."
}
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} $ps

# The kernel, from the HLS IP catalog.
set vlnv [lindex [get_ipdefs -filter "NAME == $cfg(top)"] 0]
if {$vlnv eq ""} { error "HLS IP '$cfg(top)' not in catalog $ip_repo" }
set kernel [create_bd_cell -type ip -vlnv $vlnv kernel]
set masters [get_bd_intf_pins -quiet -of $kernel -filter {MODE == Master && VLNV =~ *aximm*}]
puts "INFO: kernel $vlnv, m_axi ports: [llength $masters]"

# Which PS ports exist: GP0 (PS -> kernel registers), HP0 .. HP3 (kernel -> DDR,
# one per m_axi port, 64 bits wide), the fabric interrupt, and FCLK_CLK0 at the
# HLS target.
set hp_ports [llength $masters]
if {$hp_ports > 4} { error "the kernel has $hp_ports m_axi ports; the PS has 4 HP ports" }
set ps_config [list \
    CONFIG.PCW_USE_M_AXI_GP0 1 \
    CONFIG.PCW_USE_FABRIC_INTERRUPT 1 \
    CONFIG.PCW_IRQ_F2P_INTR 1 \
    CONFIG.PCW_EN_CLK0_PORT 1 \
    CONFIG.PCW_EN_RST0_PORT 1 \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [format %.3f [expr {1000.0 / $cfg(clock_ns)}]] \
]
for {set i 0} {$i < 4} {incr i} {
    lappend ps_config CONFIG.PCW_USE_S_AXI_HP$i [expr {$i < $hp_ports}]
    if {$i < $hp_ports} { lappend ps_config CONFIG.PCW_S_AXI_HP${i}_DATA_WIDTH 64 }
}
set_property -dict $ps_config $ps

# The PLL rarely hits the requested frequency exactly. The kernel must not run
# faster than the period HLS scheduled it for.
set act_mhz [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ $ps]
set act_ns  [expr {1000.0 / $act_mhz}]
puts "INFO: HLS target $cfg(clock_ns) ns, actual FCLK_CLK0 [format %.3f $act_ns] ns ($act_mhz MHz)"
if {$act_ns < $cfg(clock_ns) - 0.01} {
    error "FCLK_CLK0 ($act_ns ns) is faster than the HLS target ($cfg(clock_ns) ns)"
}

# Reset: turns FCLK_RESET0_N into a reset synchronous to FCLK_CLK0.
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst]

# Control path: GP0 (AXI3) to the kernel's AXI-Lite registers.
set ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect ctrl]
set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1] $ctrl

# Data path: each kernel m_axi port through its own SmartConnect (it adapts
# width and AXI version) to its own HP port: master i -> data<i> -> HP<i>.
for {set i 0} {$i < $hp_ports} {incr i} {
    set data [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect data$i]
    set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1] $data
}

# ---------------------------------------------------------------- Wires
# Clock: FCLK_CLK0 drives everything in the fabric, and the PS side of each port.
set clocked [list rst/slowest_sync_clk kernel/ap_clk ctrl/aclk ps/M_AXI_GP0_ACLK]
for {set i 0} {$i < $hp_ports} {incr i} { lappend clocked data$i/aclk ps/S_AXI_HP${i}_ACLK }
foreach pin $clocked { connect_bd_net [get_bd_pins ps/FCLK_CLK0] [get_bd_pins $pin] }

# Reset.
connect_bd_net [get_bd_pins ps/FCLK_RESET0_N] [get_bd_pins rst/ext_reset_in]
set reset [list kernel/ap_rst_n ctrl/aresetn]
for {set i 0} {$i < $hp_ports} {incr i} { lappend reset data$i/aresetn }
foreach pin $reset { connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $pin] }

# Control path.
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_GP0] [get_bd_intf_pins ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ctrl/M00_AXI] [get_bd_intf_pins kernel/s_axi_control]

# Data path (masters in name order: m_axi_gmem0 -> HP0, m_axi_gmem1 -> HP1, ...).
set i 0
foreach master [lsort $masters] {
    puts "INFO: [get_property NAME $master] -> data$i -> S_AXI_HP$i"
    connect_bd_intf_net $master [get_bd_intf_pins data$i/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins data$i/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP$i]
    incr i
}

# Interrupt (HLS adds the pin for a kernel with s_axilite port=return).
if {[llength [get_bd_pins -quiet kernel/interrupt]]} {
    connect_bd_net [get_bd_pins kernel/interrupt] [get_bd_pins ps/IRQ_F2P]
}

# ---------------------------------------------------------------- Addresses
# The kernel's registers in the PS's address map (GP0: 0x4000_0000 up), and all
# of DDR in the kernel's m_axi address map.
assign_bd_address
foreach seg [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces ps/Data]] {
    puts "INFO: PS sees [get_property NAME $seg] at [get_property OFFSET $seg]"
}

# ---------------------------------------------------------------- Finish
validate_bd_design
save_bd_design
set bd_file [get_files system.bd]
generate_target all $bd_file
add_files -norecurse [make_wrapper -files $bd_file -top]
set_property top system_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1

# DDR and FIXED_IO are the PS's own pins; the PL uses no pins, so no XDC is
# needed. If you add PL pins, put them in boards/zynq7030/system.xdc.
set xdc [file join $cfg(root) boards zynq7030 system.xdc]
if {[file exists $xdc]} { add_files -fileset constrs_1 -norecurse $xdc }
