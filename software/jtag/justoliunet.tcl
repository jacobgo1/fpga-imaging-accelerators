# Run the justoliunet kernel (1D-Justo-LiuNet pixel classifier) over JTAG.
#
# From a shell on the machine the board is plugged into:
#   xsdb software/jtag/justoliunet.tcl artifacts/justoliunet/<run>
# runs every pixel in tb/data/justoliunet_vectors.txt on the FPGA and compares
# the logits with the reference computed from the same checkpoint.
#
# Live, from the xsdb prompt:
#   xsdb% source software/jtag/justoliunet.tcl
#   xsdb% board_open artifacts/justoliunet/<run>
#   xsdb% jl_vector 3                    ;# one test pixel, shown in full
#   xsdb% jl_classify_file my_pixel.txt  ;# BANDS numbers from any source
#   xsdb% jl_classify {0.1 0.2 ...}

source [file join [file dirname [info script]] board.tcl]

set jl_vectors_file [file normalize [file join [file dirname [info script]] .. .. tb data justoliunet_vectors.txt]]

# The kernel works in float32; AXI-Lite registers carry the raw bits.
proc jl_bits {x} {
    binary scan [binary format r $x] iu bits
    return $bits
}

proc jl_float {bits} {
    binary scan [binary format i $bits] r x
    return $x
}

proc jl_bands {} {
    lassign [board::array_info spectrum] offset width depth
    return $depth
}

proc jl_run {spectrum} {
    if {[llength $spectrum] != [jl_bands]} {
        error "the kernel takes [jl_bands] bands, got [llength $spectrum]"
    }
    set words {}
    foreach x $spectrum { lappend words [jl_bits $x] }
    kernel_write spectrum $words
    kernel_run
    set logits {}
    foreach w [kernel_read logits] { lappend logits [jl_float $w] }
    return $logits
}

proc jl_argmax {values} {
    set best 0
    for {set i 1} {$i < [llength $values]} {incr i} {
        if {[lindex $values $i] > [lindex $values $best]} { set best $i }
    }
    return $best
}

proc jl_format {values} {
    set out {}
    foreach v $values { lappend out [format %10.5f $v] }
    return [join $out " "]
}

proc jl_classify {spectrum} {
    set logits [jl_run $spectrum]
    puts "logits [jl_format $logits]  -> class [jl_argmax $logits]"
    return [jl_argmax $logits]
}

proc jl_classify_file {path} {
    set f [open $path]; set text [read $f]; close $f
    return [jl_classify [regexp -all -inline {[^\s,;]+} $text]]
}

# Each vector: {inputs expected_logits}
proc jl_load_vectors {{path ""}} {
    if {$path eq ""} { set path $::jl_vectors_file }
    set f [open $path]
    set vectors {}
    while {[gets $f line] >= 0} {
        if {[string match #* $line] || [string trim $line] eq ""} continue
        lassign [split $line |] inputs expected
        lappend vectors [list [string trim $inputs] [string trim $expected]]
    }
    close $f
    return $vectors
}

proc jl_close {got want} {
    expr {abs($got - $want) <= 1e-4 + 1e-4 * abs($want)}
}

# Runs one vector; returns 1 if the FPGA agrees with the reference.
proc jl_check {inputs expected {verbose 0}} {
    set got [jl_run $inputs]
    set ok [expr {[jl_argmax $got] == [jl_argmax $expected]}]
    foreach g $got w $expected { if {![jl_close $g $w]} { set ok 0 } }
    if {$verbose || !$ok} {
        puts "  FPGA logits      [jl_format $got]  -> class [jl_argmax $got]"
        puts "  reference logits [jl_format $expected]  -> class [jl_argmax $expected]"
    }
    return $ok
}

proc jl_vector {index {path ""}} {
    set vectors [jl_load_vectors $path]
    lassign [lindex $vectors $index] inputs expected
    if {$inputs eq ""} { error "no vector $index; there are [llength $vectors]" }
    puts "pixel $index: first bands [lrange $inputs 0 5] ..."
    set ok [jl_check $inputs $expected 1]
    puts [expr {$ok ? "MATCH" : "MISMATCH"}]
    return $ok
}

# Returns 0 when every vector matches, so it can be an exit code.
proc jl_selftest {{path ""}} {
    set vectors [jl_load_vectors $path]
    set failed 0
    set index 0
    foreach v $vectors {
        lassign $v inputs expected
        set ok [jl_check $inputs $expected]
        puts [format "%s pixel %2d  class %d" [expr {$ok ? "PASS" : "FAIL"}] $index [jl_argmax $expected]]
        if {!$ok} { incr failed }
        incr index
    }
    set total [llength $vectors]
    puts [expr {$failed ? "$failed of $total pixels FAILED" : "PASS: all $total pixels match the reference"}]
    return [expr {$failed != 0}]
}

if {[info exists argv] && [llength $argv] > 0} {
    board_open {*}$argv
    exit [jl_selftest]
}
