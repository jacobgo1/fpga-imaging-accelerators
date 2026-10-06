# Vitis HLS: C simulation, C++ -> Verilog, co-simulation, IP export.
# main.py runs this in the run folder, after writing settings.tcl there.
# Each stage does what the ones before it do, then its own step:
#   csim       csim_design     the testbench, compiled by HLS
#   csynth     csynth_design   C++ -> Verilog, with the latency/resource report
#   cosim      cosim_design    the testbench driving the generated Verilog
#   bitstream  export_design   the Verilog as an IP block for Vivado (cosim first, unless skipped)
if {[catch {
    source settings.tcl
    # A project named "hls" in this folder; its one solution is called "solution".
    open_project -reset hls
    set_top $cfg(top)
    set flags "-std=c++14"
    foreach dir $cfg(include_dirs) { append flags " -I$dir" }
    foreach src $cfg(sources) { add_files $src -cflags $flags }
    foreach tb $cfg(testbench) { add_files -tb $tb -cflags $flags }
    open_solution -reset solution -flow_target vivado
    set_part $cfg(part)
    create_clock -period $cfg(clock_ns) -name default
    if {$cfg(directives) ne ""} { source $cfg(directives) }

    csim_design -clean
    if {$cfg(stage) ne "csim"} {
        csynth_design
        # Some failures ("Pre-synthesis failed") only print an ERROR and return
        # normally; without this check the flow would go on without RTL.
        set report [file join $cfg(run_dir) hls solution syn report $cfg(top)_csynth.rpt]
        if {![file exists $report]} { error "C synthesis failed: no $report" }
    }
    if {$cfg(stage) eq "cosim" || ($cfg(stage) eq "bitstream" && !$cfg(skip_cosim))} {
        cosim_design -rtl verilog -tool xsim -trace_level port
    }
    if {$cfg(stage) eq "bitstream"} {
        # Writes the IP to hls/solution/impl/ip/, where vivado.tcl picks it up.
        export_design -format ip_catalog -rtl verilog
    }
    close_project
} message options]} {
    puts stderr $message
    # Vitis HLS raises some errors without -errorinfo; do not fail while reporting.
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
