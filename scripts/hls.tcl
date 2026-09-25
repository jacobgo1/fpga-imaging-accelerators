# Invoked in an isolated run directory by main.py.
if {[catch {
    source settings.tcl
    # open_project takes a project NAME, not a path: ':' and '/' are rejected.
    # The tool already runs with this run directory as its working directory,
    # so the project lands in $cfg(run_dir)/hls either way.
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
    if {$cfg(stage) ne "csim"} { csynth_design }
    # Skipping trusts csim for behavior; the RTL itself is then only checked on hardware.
    if {$cfg(stage) in {cosim export synth impl bitstream} && !$cfg(skip_cosim)} {
        cosim_design -rtl verilog -tool xsim -trace_level port
    }
    if {$cfg(stage) in {export synth impl bitstream}} {
        file mkdir [file join $cfg(run_dir) deliverables ip]
        export_design -format ip_catalog -rtl verilog \
            -output [file join $cfg(run_dir) deliverables ip $cfg(top).zip]
    }
    close_project
} message options]} {
    puts stderr $message
    # Vitis HLS raises some errors without -errorinfo; do not fail while reporting.
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
