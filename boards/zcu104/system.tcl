# ZCU104 board_script (see ../README.md for the contract this fills in).
#
# Wires the exported HLS kernel's AXI-Lite control bundle to the Zynq
# UltraScale+ PS's GP0 master, and its m_axi gmem master to the PS's HP0
# slave (DDR) -- one control bus, one data path, matching the single
# bundle=gmem/bundle=control the kernel's interface pragmas use (see
# src/hls/justoliunet/justoliunet.cpp). scripts/vivado.tcl has already run
# `create_project`, pointed ip_repo_paths at the exported IP and called
# update_ip_catalog before sourcing this file -- see cfg(...)/ip_repo/
# rtl_dir in ../README.md's variable table.
#
# NOT YET RUN AGAINST REAL VIVADO (no AMD tools in the environment this was
# written in -- see the main README's "How this is built" section for the
# project's existing stance on unverified stages). Expect to fix small
# things against Vivado's own error messages, same as export/synth/impl/
# bitstream already are for every other kernel here. The two likeliest
# snags: the automation rule names below (`xilinx.com:bd_rule:zynq_ultra_ps_e`,
# `xilinx.com:bd_rule:axi4`) are correct for the 2023.1-era MPSoC IP but can
# rename between Vivado releases -- if `apply_bd_automation` errors,
# `get_bd_cells -regexp .*` after `create_bd_cell` and re-check the rule
# names for your installed catalog version, from the Tcl console, not the GUI.

create_bd_design "system"

# ---------------------------------------------------------------------
# 1. Zynq UltraScale+ MPSoC PS. apply_board_preset pulls DDR/clock/pinout
#    settings from the ZCU104 board files Vivado ships -- set board_part
#    in config/project.json (see boards/README.md) so this preset exists.
# ---------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 zynq_ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} [get_bd_cells zynq_ps]

# One AXI-Lite master (PS -> kernel control regs) and one high-performance
# AXI slave (kernel -> DDR), plus the PL fabric clock everything else here
# runs on.
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0    {1} \
    CONFIG.PSU__USE__S_AXI_HP0_FPD {1} \
    CONFIG.PSU__FPGA_PL0_ENABLE   {1} \
] [get_bd_cells zynq_ps]

# ---------------------------------------------------------------------
# 2. The exported HLS kernel. Looked up by VLNV pattern rather than a
#    hardcoded "...:1.0" so a version bump in a later HLS release doesn't
#    silently need an edit here.
# ---------------------------------------------------------------------
set hls_vlnv [get_ipdefs -filter "VLNV =~ {xilinx.com:hls:$cfg(top):*}"]
if {[llength $hls_vlnv] != 1} {
    error "Expected exactly one xilinx.com:hls:$cfg(top):* IP in the catalog, found: $hls_vlnv"
}
set kernel_cell [create_bd_cell -type ip -vlnv $hls_vlnv "${cfg(top)}_0"]

# ---------------------------------------------------------------------
# 3. Wiring. Matched by interface-pin name pattern (control bundle /
#    m_axi gmem) rather than one exact literal name, since Vitis HLS's
#    exact casing for a bundle name has moved between releases.
# ---------------------------------------------------------------------
set ctrl_pin [get_bd_intf_pins -of_objects [get_bd_cells $kernel_cell] \
    -filter {NAME =~ "*control*"}]
set gmem_pin [get_bd_intf_pins -of_objects [get_bd_cells $kernel_cell] \
    -filter {NAME =~ "*gmem*"}]
if {[llength $ctrl_pin] != 1} { error "Expected one *control* interface pin on $kernel_cell, found: $ctrl_pin" }
if {[llength $gmem_pin] != 1} { error "Expected one *gmem* interface pin on $kernel_cell, found: $gmem_pin" }

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config [list Master "/zynq_ps/M_AXI_HPM0_FPD" Clk "Auto"] $ctrl_pin
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config [list Master $gmem_pin Slave "/zynq_ps/S_AXI_HP0_FPD" Clk "Auto"] \
    [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]

validate_bd_design
save_bd_design

# ---------------------------------------------------------------------
# 4. Wrapper + top, and the clock XDC vivado.tcl's kernel-only path writes
#    for itself -- here we let the PS's own generated clock drive
#    everything, so the only constraint left to check is that
#    PSU__FPGA_PL0_ENABLE actually delivers cfg(clock_ns): confirm this
#    against the PS's IP configuration, not a separate create_clock.
# ---------------------------------------------------------------------
set wrapper [make_wrapper -files [get_files system.bd] -top]
add_files -norecurse $wrapper
update_compile_order -fileset sources_1
set_property top system_wrapper [current_fileset]
