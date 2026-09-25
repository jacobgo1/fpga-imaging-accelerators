# Program the ZCU104's PL fabric over JTAG from Vivado's Hardware Manager --
# batch mode, no GUI. Only reconfigures the programmable logic; whatever the
# PS already booted (e.g. PYNQ's Linux from the SD card) keeps running
# untouched, same as pressing "Program Device" in the GUI would.
#
# Usage: vivado -mode batch -source program-jtag.tcl -notrace \
#            -tclargs <path/to/system.bit> [hw_server_host:port]
if {$argc < 1} {
    puts stderr "Usage: vivado -mode batch -source program-jtag.tcl -tclargs <bitfile> \[hw_server_host:port\]"
    exit 1
}
set bitfile [lindex $argv 0]
set hw_server [expr {$argc >= 2 ? [lindex $argv 1] : "localhost:3121"}]

if {[catch {
    open_hw_manager
    connect_hw_server -url $hw_server
    open_hw_target

    set device [lindex [get_hw_devices -filter {PART =~ "*zu7ev*"}] 0]
    if {$device eq ""} {
        error "No xczu7ev device found on this hw_target. get_hw_devices returned: [get_hw_devices]"
    }
    current_hw_device $device
    refresh_hw_device -update_hw_probes false $device

    set_property PROGRAM.FILE $bitfile $device
    program_hw_devices $device
    puts "Programmed $device with $bitfile"

    close_hw_target
    disconnect_hw_server
    close_hw_manager
} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
