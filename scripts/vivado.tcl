# Kernel-only synthesis, or a complete board project generated from Tcl.
proc reports {directory} {
    file mkdir $directory
    report_utilization -hierarchical -file [file join $directory utilization.rpt]
    report_timing_summary -report_unconstrained -file [file join $directory timing.rpt]
    report_drc -file [file join $directory drc.rpt]
    report_power -file [file join $directory power.rpt]
}

if {[catch {
    source settings.tcl
    set_param general.maxThreads [expr {min(8, $cfg(jobs))}]
    set deliverables [file join $cfg(run_dir) deliverables]
    set rtl_dir [file join $cfg(run_dir) hls solution syn verilog]
    set ip_repo [file join $cfg(run_dir) hls solution impl ip]

    if {$cfg(board_script) eq ""} {
        create_project -in_memory -part $cfg(part)
        set rtl_files [glob -nocomplain -directory $rtl_dir *.v]
        if {[llength $rtl_files] == 0} { error "HLS generated no Verilog in $rtl_dir" }
        add_files $rtl_files
        foreach data [glob -nocomplain -directory $rtl_dir *.dat] { add_files $data }
        # HLS may emit Tcl to create arithmetic sub-IP for some operators.
        foreach script [glob -nocomplain -directory $rtl_dir *.tcl] { source $script }
        set clock_xdc [file join $cfg(run_dir) kernel_clock.xdc]
        set clock_file [open $clock_xdc w]
        puts $clock_file [format {create_clock -name ap_clk -period %s [get_ports ap_clk]} $cfg(clock_ns)]
        close $clock_file
        read_xdc $clock_xdc
        synth_design -top $cfg(top) -part $cfg(part) -mode out_of_context
        reports [file join $cfg(run_dir) vivado_reports synth]
        file mkdir [file join $deliverables checkpoints]
        write_checkpoint -force [file join $deliverables checkpoints kernel_synth.dcp]
    } else {
        create_project system [file join $cfg(run_dir) vivado] -part $cfg(part) -force
        set_property ip_repo_paths [list $ip_repo] [current_project]
        update_ip_catalog
        # Contract: populate this project; set top; add XDC; generate BD targets.
        source $cfg(board_script)
        update_compile_order -fileset sources_1
        launch_runs synth_1 -jobs $cfg(jobs)
        wait_on_run synth_1
        if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
            error "Vivado synthesis failed: [get_property STATUS [get_runs synth_1]]"
        }
        open_run synth_1
        reports [file join $cfg(run_dir) vivado_reports synth]
        file mkdir [file join $deliverables checkpoints]
        write_checkpoint -force [file join $deliverables checkpoints system_synth.dcp]
        if {$cfg(stage) in {impl bitstream}} {
            set step route_design
            if {$cfg(stage) eq "bitstream"} { set step write_bitstream }
            launch_runs impl_1 -to_step $step -jobs $cfg(jobs)
            wait_on_run impl_1
            if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
                error "Vivado implementation failed: [get_property STATUS [get_runs impl_1]]"
            }
            open_run impl_1
            reports [file join $cfg(run_dir) vivado_reports impl]
            write_checkpoint -force [file join $deliverables checkpoints system_routed.dcp]
            set paths [get_timing_paths -quiet -max_paths 1 -slack_lesser_than 0]
            set hold_paths [get_timing_paths -quiet -delay_type min -max_paths 1 -slack_lesser_than 0]
            if {[llength $paths] || [llength $hold_paths]} {
                error "Routed design fails timing; inspect timing.rpt"
            }
            if {$cfg(stage) eq "bitstream"} {
                set bits [glob -nocomplain -directory [get_property DIRECTORY [get_runs impl_1]] *.bit]
                if {[llength $bits] == 0} { error "Implementation produced no .bit file" }
                file mkdir [file join $deliverables bitstream]
                foreach bit $bits { file copy -force $bit [file join $deliverables bitstream] }
                if {[llength [get_debug_cores -quiet]] > 0} {
                    write_debug_probes -force [file join $deliverables bitstream system.ltx]
                }
                if {$cfg(export_xsa)} {
                    file mkdir [file join $deliverables hardware]
                    write_hw_platform -fixed -include_bit -force \
                        -file [file join $deliverables hardware system.xsa]
                }
            }
        }
    }
} message options]} {
    puts stderr $message
    # Some tool errors arrive without -errorinfo; do not fail while reporting.
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
