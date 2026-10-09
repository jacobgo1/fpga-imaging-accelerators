# Vivado: the block design around the HLS IP, synthesis, place and route, .bit.
# main.py runs this in the run folder after scripts/hls.tcl has exported the IP.
#
# Results, all inside the run folder:
#   vivado/system.xpr                         the project; open it in Vivado to look around
#   vivado/system.runs/impl_1/*.bit           the bitstream
#   reports/synth/, reports/impl/             utilization, timing, DRC, power

proc reports {directory} {
    file mkdir $directory
    report_utilization -hierarchical -file [file join $directory utilization.rpt]
    report_timing_summary -report_unconstrained -file [file join $directory timing.rpt]
    report_drc -file [file join $directory drc.rpt]
    report_power -file [file join $directory power.rpt]
}

proc run_and_check {run args} {
    launch_runs $run -jobs $::cfg(jobs) {*}$args
    wait_on_run $run
    if {[get_property PROGRESS [get_runs $run]] ne "100%"} {
        error "$run failed: [get_property STATUS [get_runs $run]]"
    }
    open_run $run
}

if {[catch {
    source settings.tcl
    # Cores: cfg(jobs) runs at once (launch_runs -jobs: the block design's IPs
    # are synthesized as separate runs, in parallel), each with up to 8 threads,
    # Vivado's limit. set_param holds only in the process that runs it, and
    # launch_runs starts each run as a new one: they source threads.tcl first.
    set threads [expr {min(8, $cfg(jobs))}]
    set_param general.maxThreads $threads
    set threads_tcl [file join $cfg(run_dir) threads.tcl]
    set f [open $threads_tcl w]
    puts $f "set_param general.maxThreads $threads"
    close $f

    # 1. An empty project, with the kernel's IP in its catalog.
    set ip_repo [file join $cfg(run_dir) hls solution impl ip]
    create_project system [file join $cfg(run_dir) vivado] -part $cfg(part) -force
    set_property ip_repo_paths [list $ip_repo] [current_project]
    update_ip_catalog
    set_property STEPS.SYNTH_DESIGN.TCL.PRE $threads_tcl [get_runs synth_1]
    set_property STEPS.INIT_DESIGN.TCL.PRE  $threads_tcl [get_runs impl_1]

    # 2. The block design: boards/BOARD/system.tcl. It leaves system_wrapper as the top.
    source $cfg(board_script)

    # 3. Synthesis.
    run_and_check synth_1
    reports [file join $cfg(run_dir) reports synth]

    # 4. Place, route and write the bitstream.
    run_and_check impl_1 -to_step write_bitstream
    reports [file join $cfg(run_dir) reports impl]
    if {[llength [get_timing_paths -quiet -max_paths 1 -slack_lesser_than 0]] ||
        [llength [get_timing_paths -quiet -delay_type min -max_paths 1 -slack_lesser_than 0]]} {
        error "Routed design fails timing; see reports/impl/timing.rpt"
    }
    if {![llength [glob -nocomplain -directory [get_property DIRECTORY [get_runs impl_1]] *.bit]]} {
        error "Implementation produced no .bit file"
    }
} message options]} {
    puts stderr $message
    # Some tool errors arrive without -errorinfo; do not fail while reporting.
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
