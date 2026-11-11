#*****************************************************************************************
# Add an HLS kernel to the OPU base design (opu-system's opu_base.tcl), for any kernel:
# it wires whatever m_axi ports the kernel has. Sourced by boards/opu/project.tcl, with
# projectDir, projectName, hlsTop and hlsClockNs set, the kernel's export unpacked in
# $projectDir/${projectName}_hls_export. As cnn-1d-picobob's add-cnn-accelerator.tcl:
#
#   processing_system7_0
#     FCLK_CLK0 ──► (hls_clock: MMCM, only if the kernel needs a slower clock) ──► clk
#     M_AXI_GP0 ──► axi_interconnect_0 ─ Mnn ──► hls_kernel/s_axi_control   registers
#     S_AXI_HP0 ◄── hls_data0 (SmartConnect) ◄── hls_kernel/m_axi_gmem0     DDR access,
#     ...                                   ...                          one HP port per
#     S_AXI_HP3 ◄── hls_data3 (SmartConnect) ◄── hls_kernel/m_axi_gmem3     m_axi port; a
#                                            ◄── hls_kernel/m_axi_gmem4     fifth shares HP3
#
# The base design (UART, timer, the bitstream-hash ROM) stays, so opu-services and
# obip fpga hash work with the kernel loaded. The block is called hls_kernel: the
# devicetree overlay (boards/opu/hls_kernel.dtsi) refers to it by that name.
#*****************************************************************************************
puts "Add HLS kernel $hlsTop"

set ps [get_bd_cells processing_system7_0]

# The kernel's IP, next to the base design's (CubeDMA's, on PicoBoB).
set ip_dir $projectDir/${projectName}_hls_export
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] [list $ip_dir]] [current_project]
update_ip_catalog -rebuild
set vlnv [lindex [get_ipdefs -filter "NAME == $hlsTop"] 0]
if {$vlnv eq ""} { error "HLS IP '$hlsTop' not in $ip_dir" }
set kernel [create_bd_cell -type ip -vlnv $vlnv hls_kernel]
set masters [lsort [get_bd_intf_pins -quiet -of $kernel -filter {MODE == Master && VLNV =~ *aximm*}]]
set n_masters [llength $masters]
puts "INFO: kernel $vlnv, m_axi ports: $n_masters"
if {$n_masters > 5} { error "the kernel has $n_masters m_axi ports; the PS has 4 HP ports (and HP3 takes a fifth)" }

#*****************************************************************************************
# The kernel's clock and reset
#*****************************************************************************************
# A bitstream loaded while Linux runs cannot change the PS's clocks, so FCLK_CLK0 stays as
# the board's boot configuration sets it (100 MHz). A kernel built for a slower clock gets
# it from an MMCM in the PL; Vivado's timing then checks the kernel at that clock.
set fclk_mhz [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ $ps]
set kernel_mhz [expr {1000.0 / $hlsClockNs}]
if {$kernel_mhz >= $fclk_mhz - 0.01} {
    puts "INFO: kernel on FCLK_CLK0 ($fclk_mhz MHz, built for $kernel_mhz MHz)"
    set clk  [get_bd_pins processing_system7_0/FCLK_CLK0]
    set rstn [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
} else {
    puts "INFO: kernel on an MMCM at [format %.3f $kernel_mhz] MHz, from FCLK_CLK0 ($fclk_mhz MHz)"
    create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz hls_clock
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins hls_clock/clk_in1]
    set_property -dict [list \
        CONFIG.CLKOUT1_REQUESTED_OUT_FREQ [format %.3f $kernel_mhz] \
        CONFIG.RESET_TYPE {ACTIVE_LOW} \
        CONFIG.RESET_PORT {resetn} \
        CONFIG.USE_LOCKED {true} \
    ] [get_bd_cells hls_clock]
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins hls_clock/resetn]
    # A reset synchronous to that clock, held until the MMCM locks.
    create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset hls_reset
    connect_bd_net [get_bd_pins hls_clock/clk_out1] [get_bd_pins hls_reset/slowest_sync_clk]
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins hls_reset/ext_reset_in]
    connect_bd_net [get_bd_pins hls_clock/locked] [get_bd_pins hls_reset/dcm_locked]
    set clk  [get_bd_pins hls_clock/clk_out1]
    set rstn [get_bd_pins hls_reset/peripheral_aresetn]
}
connect_bd_net $clk  [get_bd_pins hls_kernel/ap_clk]
connect_bd_net $rstn [get_bd_pins hls_kernel/ap_rst_n]

#*****************************************************************************************
# Control: one more master on the base design's AXI interconnect (it crosses the clocks)
#*****************************************************************************************
set n_master [get_property CONFIG.NUM_MI [get_bd_cells axi_interconnect_0]]
set_property CONFIG.NUM_MI [expr {$n_master + 1}] [get_bd_cells axi_interconnect_0]
set m [format M%02d $n_master]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/${m}_AXI] [get_bd_intf_pins hls_kernel/s_axi_control]
connect_bd_net $clk  [get_bd_pins axi_interconnect_0/${m}_ACLK]
connect_bd_net $rstn [get_bd_pins axi_interconnect_0/${m}_ARESETN]

#*****************************************************************************************
# Data: each m_axi port through its own SmartConnect to its own 64-bit HP port, in name
# order (m_axi_gmem0 -> HP0, ...). A fifth shares HP3 with the fourth: meant for a writer
# next to a reader, as AXI reads and writes have their own channels.
#*****************************************************************************************
set hp_ports [expr {min($n_masters, 4)}]
set ps_config {}
for {set i 0} {$i < $hp_ports} {incr i} {
    lappend ps_config CONFIG.PCW_USE_S_AXI_HP$i 1 CONFIG.PCW_S_AXI_HP${i}_DATA_WIDTH 64
}
if {[llength $ps_config]} { set_property -dict $ps_config $ps }
for {set i 0} {$i < $hp_ports} {incr i} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect hls_data$i
    set inputs [expr {$i == 3 && $n_masters == 5 ? 2 : 1}]
    set_property -dict [list CONFIG.NUM_SI $inputs CONFIG.NUM_MI 1] [get_bd_cells hls_data$i]
    connect_bd_net $clk  [get_bd_pins hls_data$i/aclk]
    connect_bd_net $rstn [get_bd_pins hls_data$i/aresetn]
    # The HP port on the kernel's clock. opu_base.tcl ties HP0's to the GP0 clock: undone.
    set aclk [get_bd_pins processing_system7_0/S_AXI_HP${i}_ACLK]
    set net [get_bd_nets -quiet -of_objects $aclk]
    if {$net ne ""} { disconnect_bd_net $net $aclk }
    connect_bd_net $clk $aclk
    connect_bd_intf_net [get_bd_intf_pins hls_data$i/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP$i]
}
set i 0
foreach master $masters {
    if {$i < 4} {
        puts "INFO: [get_property NAME $master] -> hls_data$i -> S_AXI_HP$i"
        connect_bd_intf_net $master [get_bd_intf_pins hls_data$i/S00_AXI]
    } else {
        puts "INFO: [get_property NAME $master] -> hls_data3 (shared) -> S_AXI_HP3"
        connect_bd_intf_net $master [get_bd_intf_pins hls_data3/S01_AXI]
    }
    incr i
}

#*****************************************************************************************
# Assign addresses: the kernel's registers for the PS, all of DDR for the kernel
#*****************************************************************************************
assign_bd_address

#*****************************************************************************************
# Reports after routing, next to the hardware files: what the kernel uses, and its timing
#*****************************************************************************************
set sdkDir $projectDir/$projectName/$projectName.sdk
file mkdir $sdkDir
set hook $projectDir/$projectName/hls_reports.tcl
set f [open $hook w]
puts $f "report_utilization -hierarchical -file {$sdkDir/utilization.rpt}"
puts $f "report_timing_summary -file {$sdkDir/timing.rpt}"
close $f
set_property STEPS.ROUTE_DESIGN.TCL.POST $hook [get_runs impl_1]
