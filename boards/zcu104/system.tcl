# boards/zcu104/system.tcl -- the hardware around the HLS kernel, wire by wire.
#
# scripts/vivado.tcl sources this after it has created an empty project and
# put the exported HLS IP in its catalog. Everything this design contains is
# created and connected below; nothing is added automatically.
#
#   PS ("ps": 4 x ARM, DDR controller)
#     pl_clk0  ───────────────────────────────┬──► every clock pin below
#     pl_resetn0 ──► rst (proc_sys_reset) ────┴──► every reset pin below
#     M_AXI_HPM0_FPD ──► ctrl (SmartConnect)  ──► kernel/s_axi_control   registers
#     S_AXI_HP0_FPD  ◄── data0 (SmartConnect) ◄── kernel/m_axi_gmem0     DDR access
#     S_AXI_HP1_FPD  ◄── data1 (SmartConnect) ◄── kernel/m_axi_gmem1     (one port to DDR
#     ...                                         ...                    per m_axi port)
#     S_AXI_HPC0_FPD ◄── data4 (SmartConnect) ◄── kernel/m_axi_gmem4     (the 5th and 6th
#     S_AXI_HPC1_FPD ◄── data5 (SmartConnect) ◄── kernel/m_axi_gmem5      on the HPC ports)
#     pl_ps_irq0     ◄── kernel/interrupt                                (unused by software)
#
# The data path only exists if the kernel has m_axi ports. Each gets a port to
# DDR of its own (up to six), 128 bits per cycle each: first the four HP ports,
# then the two HPC ports. HPC can be cache coherent, but nothing here turns that
# on, so they work like the HP ports (the notebook flushes and invalidates its
# buffers either way). justounetsimple_opt reads its input through four ports at
# once (gmem0-3, HP0-3) and writes through a fifth (gmem4, HPC0). A kernel with
# only s_axilite (matmul) gets the control path alone.
#
# To change the design, edit this file. To look at what it produced, open
# build/KERNEL/latest/vivado/system.xpr in Vivado and open the block design.

# ---------------------------------------------------------------- Project
set bp [lindex [get_board_parts -quiet -latest_file_version *zcu104*] 0]
if {$bp eq ""} { error "ZCU104 board files not found. Check with: get_board_parts *zcu104*" }
set_property board_part $bp [current_project]
create_bd_design system

# ---------------------------------------------------------------- Blocks
# The PS. The board preset only configures the chip's own pins and DDR for the
# ZCU104 (MIO, DDR timing); it creates no blocks and no wires.
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset 1} $ps

# The kernel, from the HLS IP catalog.
set vlnv [lindex [get_ipdefs -filter "NAME == $cfg(top)"] 0]
if {$vlnv eq ""} { error "HLS IP '$cfg(top)' not in catalog $ip_repo" }
set kernel [create_bd_cell -type ip -vlnv $vlnv kernel]
set masters [get_bd_intf_pins -quiet -of $kernel -filter {MODE == Master && VLNV =~ *aximm*}]
puts "INFO: kernel $vlnv, m_axi ports: [llength $masters]"

# The PS's ports to DDR, in the order the m_axi ports get them: for each, its
# switch in the PS configuration (the PS calls HP0-3 S_AXI_GP2-5 and HPC0-1
# S_AXI_GP0-1), its interface and its clock pin.
set ddr_ports {
    {S_AXI_GP2 S_AXI_HP0_FPD  saxihp0_fpd_aclk}
    {S_AXI_GP3 S_AXI_HP1_FPD  saxihp1_fpd_aclk}
    {S_AXI_GP4 S_AXI_HP2_FPD  saxihp2_fpd_aclk}
    {S_AXI_GP5 S_AXI_HP3_FPD  saxihp3_fpd_aclk}
    {S_AXI_GP0 S_AXI_HPC0_FPD saxihpc0_fpd_aclk}
    {S_AXI_GP1 S_AXI_HPC1_FPD saxihpc1_fpd_aclk}
}

# Which PS ports exist: HPM0_FPD (PS -> kernel registers), one port to DDR per
# m_axi port, one interrupt, and the PL clock at the HLS target.
set hp_ports [llength $masters]
if {$hp_ports > [llength $ddr_ports]} {
    error "the kernel has $hp_ports m_axi ports; the PS has 6 ports to DDR (4 HP, 2 HPC)"
}
set ps_config [list \
    CONFIG.PSU__USE__M_AXI_GP0 1 \
    CONFIG.PSU__USE__M_AXI_GP1 0 \
    CONFIG.PSU__USE__M_AXI_GP2 0 \
    CONFIG.PSU__USE__IRQ0      1 \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ [format %.3f [expr {1000.0 / $cfg(clock_ns)}]] \
]
for {set i 0} {$i < [llength $ddr_ports]} {incr i} {
    lappend ps_config CONFIG.PSU__USE__[lindex $ddr_ports $i 0] [expr {$i < $hp_ports}]
}
set_property -dict $ps_config $ps

# The PLL rarely hits the requested frequency exactly. The kernel must not run
# faster than the period HLS scheduled it for.
set act_mhz [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
set act_ns  [expr {1000.0 / $act_mhz}]
puts "INFO: HLS target $cfg(clock_ns) ns, actual pl_clk0 [format %.3f $act_ns] ns ($act_mhz MHz)"
if {$act_ns < $cfg(clock_ns) - 0.01} {
    error "pl_clk0 ($act_ns ns) is faster than the HLS target ($cfg(clock_ns) ns)"
}

# Reset: turns the PS's pl_resetn0 into a reset synchronous to pl_clk0.
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst]

# Control path: one AXI-Lite link from the PS to the kernel's registers.
set ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect ctrl]
set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1] $ctrl

# Data path: each kernel m_axi port through its own SmartConnect (it adapts
# the port's width and AXI details) to its own port to DDR, so the ports do not
# share one port's bandwidth: master i -> data<i> -> ddr_ports i.
for {set i 0} {$i < $hp_ports} {incr i} {
    set data [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect data$i]
    set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1] $data
}

# ---------------------------------------------------------------- Wires
# Clock: pl_clk0 drives everything in the fabric, and the PS side of each port.
set clocked [list rst/slowest_sync_clk kernel/ap_clk ctrl/aclk ps/maxihpm0_fpd_aclk]
for {set i 0} {$i < $hp_ports} {incr i} { lappend clocked data$i/aclk ps/[lindex $ddr_ports $i 2] }
foreach pin $clocked { connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins $pin] }

# Reset.
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst/ext_reset_in]
set reset [list kernel/ap_rst_n ctrl/aresetn]
for {set i 0} {$i < $hp_ports} {incr i} { lappend reset data$i/aresetn }
foreach pin $reset { connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $pin] }

# Control path.
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ctrl/M00_AXI]      [get_bd_intf_pins kernel/s_axi_control]

# Data path (masters in name order: m_axi_gmem0 -> HP0, m_axi_gmem1 -> HP1, ...,
# m_axi_gmem4 -> HPC0).
set i 0
foreach master [lsort $ ] {
    set port [lindex $ddr_ports $i 1]
    puts "INFO: [get_property NAME $master] -> data$i -> $port"
    connect_bd_intf_net $master [get_bd_intf_pins data$i/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins data$i/M00_AXI] [get_bd_intf_pins ps/$port]
    incr i
}

# Interrupt (HLS adds the pin for a kernel with s_axilite port=return).
if {[llength [get_bd_pins -quiet kernel/interrupt]]} {
    connect_bd_net [get_bd_pins kernel/interrupt] [get_bd_pins ps/pl_ps_irq0]
}

# ---------------------------------------------------------------- Addresses
# The kernel's registers in the PS's address map (0xA000_0000 on the ZCU104),
# and all of DDR in the kernel's m_axi address map. PYNQ reads both from the .hwh.
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

# No external pins, so no XDC is needed (pl_clk0 is constrained by the PS IP).
# If you add LEDs or PMOD pins, put them in boards/zcu104/system.xdc.
set xdc [file join $cfg(root) boards zcu104 system.xdc]
if {[file exists $xdc]} { add_files -fileset constrs_1 -norecurse $xdc }
