# Matrix-level helpers for the matmul kernel on top of board.tcl.
#
# One-shot self-test from a shell on the machine the board is plugged into:
#   xsdb software/jtag/matmul.tcl artifacts/matmul/<run>
#
# Live, from the xsdb prompt:
#   xsdb% source software/jtag/matmul.tcl
#   xsdb% board_open artifacts/matmul/<run>
#   xsdb% mm_print [mm_multiply [mm_identity] [mm_random]]
#   xsdb% mm_selftest 20

source [file join [file dirname [info script]] board.tcl]

proc mm_size {} {
    lassign [board::array_info a] offset width depth
    return [expr {int(sqrt($depth))}]
}

# Matrices are lists of rows: {{1 2} {3 4}}.
proc mm_multiply {a b} {
    kernel_write a [concat {*}$a]
    kernel_write b [concat {*}$b]
    kernel_run
    set flat [kernel_read c -signed]
    set n [mm_size]
    set rows {}
    for {set i 0} {$i < $n} {incr i} { lappend rows [lrange $flat [expr {$i * $n}] [expr {$i * $n + $n - 1}]] }
    return $rows
}

proc mm_reference {a b} {
    set n [llength $a]
    set rows {}
    for {set i 0} {$i < $n} {incr i} {
        set row {}
        for {set j 0} {$j < $n} {incr j} {
            set sum 0
            for {set k 0} {$k < $n} {incr k} {
                set sum [expr {$sum + [lindex $a $i $k] * [lindex $b $k $j]}]
            }
            lappend row $sum
        }
        lappend rows $row
    }
    return $rows
}

proc mm_fill {script} {
    set n [mm_size]
    set rows {}
    for {set i 0} {$i < $n} {incr i} {
        set row {}
        for {set j 0} {$j < $n} {incr j} { lappend row [apply [list {i j} $script] $i $j] }
        lappend rows $row
    }
    return $rows
}

proc mm_identity {} { mm_fill {expr {$i == $j}} }
proc mm_random {{lo -16} {hi 16}} { mm_fill "expr {$lo + int(rand() * ($hi - $lo + 1))}" }
proc mm_constant {v} { mm_fill "expr {$v}" }

proc mm_print {m} {
    foreach row $m {
        set cells {}
        foreach v $row { lappend cells [format %8s $v] }
        puts [join $cells " "]
    }
}

# Returns 0 on success so it can be an exit code.
proc mm_selftest {{random_cases 10}} {
    set cases [list \
        zero     [mm_constant 0]      [mm_random] \
        identity [mm_random]          [mm_identity] \
        extremes [mm_constant -32768] [mm_constant 32767]]
    for {set t 0} {$t < $random_cases} {incr t} {
        lappend cases "random$t" [mm_random -32768 32767] [mm_random -32768 32767]
    }
    set failed 0
    foreach {name a b} $cases {
        set got [mm_multiply $a $b]
        set want [mm_reference $a $b]
        if {$got eq $want} {
            puts "PASS $name"
        } else {
            incr failed
            puts "FAIL $name"
            puts "A:"; mm_print $a
            puts "B:"; mm_print $b
            puts "expected:"; mm_print $want
            puts "got:"; mm_print $got
        }
    }
    puts [expr {$failed ? "$failed of [expr {[llength $cases] / 3}] cases FAILED" : "PASS: all [expr {[llength $cases] / 3}] cases match"}]
    return [expr {$failed != 0}]
}

if {[info exists argv] && [llength $argv] > 0} {
    board_open {*}$argv
    exit [mm_selftest]
}
