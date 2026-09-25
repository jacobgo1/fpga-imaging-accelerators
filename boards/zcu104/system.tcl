# boards/zcu104/system.tcl
# Headless board integration for ZCU104 (Zynq UltraScale+ MPSoC).
# Sourced by scripts/vivado.tcl after the project and HLS IP catalog exist.
# Assumes the HLS kernel exposes s_axi_control (AXI-Lite) and zero or more
# m_axi_* masters. Adjust the "Kernel" section if it uses AXI-Stream instead.

# ---------------------------------------------------------------- Board preset
set bp [lindex [get_board_parts -quiet -latest_file_version *zcu104*] 0]
if {$bp eq ""} {
    error "ZCU104 board files not found. Check with: get_board_parts *zcu104*"
}
set_property board_part $bp [current_project]

# ---------------------------------------------------------------- Block design
set bd system
create_bd_design $bd

# PS with board presets (DDR, MIO, etc.)
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} $ps

# PL clock at the HLS target; one control master (HPM0_FPD),
# one high-performance slave (HP0_FPD = S_AXI_GP2), one PL->PS interrupt.
set target_mhz [expr {1000.0 / $cfg(clock_ns)}]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__IRQ0      {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ [format %.3f $target_mhz] \
] $ps

# ---------------------------------------------------------------- Clock check
# The PLL divisors rarely hit the request exactly. The kernel must not run
# faster than the period HLS scheduled for.
set act_mhz [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
set act_ns  [expr {1000.0 / $act_mhz}]
puts "INFO: HLS target $cfg(clock_ns) ns, actual pl_clk0 [format %.3f $act_ns] ns ($act_mhz MHz)"
if {$act_ns < $cfg(clock_ns) - 0.01} {
    error "pl_clk0 ($act_ns ns) is faster than the HLS target ($cfg(clock_ns) ns)"
}

# ---------------------------------------------------------------- Kernel
set vlnv [lindex [get_ipdefs -filter "NAME == $cfg(top)"] 0]
if {$vlnv eq ""} { error "HLS IP '$cfg(top)' not in catalog $ip_repo" }
set k [create_bd_cell -type ip -vlnv $vlnv kernel]

# Control: PS HPM0 -> kernel s_axi_control (interconnect, clock and reset auto)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Master /ps/M_AXI_HPM0_FPD Slave /kernel/s_axi_control \
    Clk_master Auto Clk_slave Auto Clk_xbar Auto \
    intc_ip {New AXI SmartConnect} master_apm 0 \
] [get_bd_intf_pins kernel/s_axi_control]

# Data: every kernel m_axi -> PS HP0 (only enabled if the kernel has masters)
set masters [get_bd_intf_pins -quiet -of $k -filter {MODE == Master && VLNV =~ *aximm*}]
if {[llength $masters]} {
    set_property CONFIG.PSU__USE__S_AXI_GP2 {1} $ps
    foreach m $masters {
        apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
            Master $m Slave /ps/S_AXI_HP0_FPD \
            Clk_master Auto Clk_slave Auto Clk_xbar Auto \
            intc_ip Auto master_apm 0 \
        ] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
    }
} else {
    puts "INFO: kernel has no m_axi masters; HP0 left disabled"
}

# Interrupt (ap_ctrl_hs + s_axilite gives an 'interrupt' pin)
if {[llength [get_bd_pins -quiet kernel/interrupt]]} {
    connect_bd_net [get_bd_pins kernel/interrupt] [get_bd_pins ps/pl_ps_irq0]
}

# ---------------------------------------------------------------- Finalize
assign_bd_address
validate_bd_design
save_bd_design

set bd_file [get_files $bd.bd]
generate_target all $bd_file
add_files -norecurse [make_wrapper -files $bd_file -top]

set_property top ${bd}_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1

# ---------------------------------------------------------------- Constraints
# pl_clk0 is constrained by the PS IP itself. This design has no PL pins, so
# no XDC is needed. If you add external ports (LEDs, PMOD), add them here:
set xdc [file join $cfg(root) boards zcu104 system.xdc]
if {[file exists $xdc]} { add_files -fileset constrs_1 -norecurse $xdc }