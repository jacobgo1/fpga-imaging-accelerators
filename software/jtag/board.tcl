# Drive an HLS kernel on a Zynq UltraScale+ board over JTAG with xsdb, the
# debugger that ships with Vivado. Works for any kernel whose arrays sit on
# its s_axilite bundle; offsets come from the HLS driver header.
#
#   xsdb% source software/jtag/board.tcl
#   xsdb% board_open artifacts/matmul/<run>      ;# program + start the PS
#   xsdb% kernel_write a {1 2 3 ...}
#   xsdb% kernel_run
#   xsdb% kernel_read c -signed

namespace eval board {
    variable base ""
    variable ctrl ""
    variable arrays    ;# name -> {offset width depth}
    array set arrays {}
}

proc board_open {artifact_dir {base ""}} {
    set bits [glob -nocomplain -directory [file join $artifact_dir bitstream] *.bit]
    if {[llength $bits] != 1} { error "expected one .bit in $artifact_dir/bitstream, found [llength $bits]" }
    set psu_init [file join $artifact_dir board psu_init.tcl]
    if {![file exists $psu_init]} { error "missing $psu_init (see boards/zcu104/system.tcl)" }
    board_load_map $artifact_dir $base
    board::bringup [lindex $bits 0] $psu_init
    puts "Ready: kernel at [format 0x%08X $board::base], arrays: [lsort [array names board::arrays]]"
}

# Reads the register map only; no hardware access.
proc board_load_map {artifact_dir {base ""}} {
    set headers [glob -nocomplain -directory [file join $artifact_dir board] *_hw.h]
    if {[llength $headers] != 1} { error "expected one *_hw.h in $artifact_dir/board, found [llength $headers]" }
    if {$base eq ""} {
        set address [file join $artifact_dir board address.tcl]
        if {![file exists $address]} {
            error "missing $address; pass the kernel base from the Address Editor: board_open <dir> 0xA0000000"
        }
        namespace eval ::board::address [list source $address]
        set base $::board::address::kernel_base
    }
    set board::base [expr {$base}]
    board::parse_header [lindex $headers 0]
}

proc board::parse_header {path} {
    variable ctrl
    variable arrays
    set f [open $path]; set text [read $f]; close $f
    array unset arrays
    if {![regexp {#define\s+X\w*?_ADDR_AP_CTRL\s+(0x[0-9a-fA-F]+)} $text -> ctrl]} {
        error "$path has no AP_CTRL register"
    }
    foreach {-> name offset} [regexp -all -inline {#define\s+X\w*?_ADDR_(\w+)_BASE\s+(0x[0-9a-fA-F]+)} $text] {
        if {![regexp "#define\\s+X\\w*?_ADDR_${name}_HIGH\\s+(0x\[0-9a-fA-F\]+)" $text -> high]
            || ![regexp "#define\\s+X\\w*?_WIDTH_${name}\\s+(\\d+)" $text -> width]
            || ![regexp "#define\\s+X\\w*?_DEPTH_${name}\\s+(\\d+)" $text -> depth]} {
            error "$path: incomplete definition for array $name"
        }
        set needed [expr {[board::words $width $depth] * 4}]
        set span [expr {$high - $offset + 1}]
        if {$needed > $span} {
            error "$path: $name needs $needed bytes but its window is $span; packing assumption is wrong"
        }
        set arrays([string tolower $name]) [list $offset $width $depth]
    }
}

# HLS packs narrow elements into 32-bit words (low element in the low bits)
# and splits wide ones over consecutive words, low word first.
proc board::slot {width} {
    foreach bits {8 16 32} { if {$width <= $bits} { return $bits } }
    return [expr {($width + 31) / 32 * 32}]
}

proc board::words {width depth} {
    return [expr {($depth * [slot $width] + 31) / 32}]
}

proc board::pack {values width} {
    set slot [slot $width]
    set mask [expr {(1 << $width) - 1}]
    set words {}
    if {$slot <= 32} {
        set per [expr {32 / $slot}]
        for {set i 0} {$i < [llength $values]} {incr i $per} {
            set word 0
            for {set j 0} {$j < $per && $i + $j < [llength $values]} {incr j} {
                set word [expr {$word | (([lindex $values [expr {$i + $j}]] & $mask) << ($j * $slot))}]
            }
            lappend words $word
        }
    } else {
        foreach v $values {
            set v [expr {$v & $mask}]
            for {set k 0} {$k < $slot / 32} {incr k} {
                lappend words [expr {($v >> (32 * $k)) & 0xFFFFFFFF}]
            }
        }
    }
    return $words
}

proc board::unpack {words width depth signed} {
    set slot [slot $width]
    set mask [expr {(1 << $width) - 1}]
    set values {}
    for {set i 0} {$i < $depth} {incr i} {
        if {$slot <= 32} {
            set per [expr {32 / $slot}]
            set word [lindex $words [expr {$i / $per}]]
            set v [expr {($word >> (($i % $per) * $slot)) & $mask}]
        } else {
            set n [expr {$slot / 32}]
            set v 0
            for {set k 0} {$k < $n} {incr k} {
                set v [expr {$v | (([lindex $words [expr {$i * $n + $k}]] & 0xFFFFFFFF) << (32 * $k))}]
            }
            set v [expr {$v & $mask}]
        }
        if {$signed && $v >> ($width - 1)} { set v [expr {$v - (1 << $width)}] }
        lappend values $v
    }
    return $values
}

proc board::array_info {name} {
    variable arrays
    if {![info exists arrays($name)]} {
        error "no s_axilite array '$name'; have: [lsort [array names arrays]]"
    }
    return $arrays($name)
}

proc kernel_write {name values} {
    lassign [board::array_info $name] offset width depth
    if {[llength $values] != $depth} { error "$name holds $depth values, got [llength $values]" }
    set words {}
    foreach w [board::pack $values $width] { lappend words [format 0x%08X $w] }
    mwr -force [expr {$board::base + $offset}] $words
}

proc kernel_read {name {signed ""}} {
    lassign [board::array_info $name] offset width depth
    set words [mrd -force -value [expr {$board::base + $offset}] [board::words $width $depth]]
    return [board::unpack $words $width $depth [expr {$signed eq "-signed"}]]
}

proc kernel_status {} {
    set v [mrd -force -value [expr {$board::base + $board::ctrl}]]
    return [list start [expr {$v & 1}] done [expr {($v >> 1) & 1}] idle [expr {($v >> 2) & 1}]]
}

# ap_done is clear-on-read, so the first read that sees it is the only one.
proc kernel_run {{timeout_ms 2000}} {
    set ctrl [expr {$board::base + $board::ctrl}]
    mwr -force $ctrl 0x1
    set deadline [expr {[clock milliseconds] + $timeout_ms}]
    while {!([mrd -force -value $ctrl] & 0x2)} {
        if {[clock milliseconds] > $deadline} {
            error "kernel did not finish within $timeout_ms ms: [kernel_status]"
        }
        after 1
    }
}

# Same sequence as the Vitis-generated JTAG debug scripts: reset, program the
# PL, then run the PS init that starts pl_clk0, releases the PL resets and
# opens the PS->PL AXI ports. Without it the kernel has no clock.
proc board::bringup {bit psu_init} {
    connect
    targets -set -nocase -filter {name =~ "*PSU*"}
    rst -system
    after 1000
    fpga -file $bit
    targets -set -nocase -filter {name =~ "*A53*#0"}
    rst -processor
    uplevel #0 [list source $psu_init]
    uplevel #0 psu_init
    after 1000
    uplevel #0 psu_ps_pl_isolation_removal
    after 1000
    uplevel #0 psu_ps_pl_reset_config
    targets -set -nocase -filter {name =~ "*A53*#0"}
    rst -processor
}
