# Matrix-level helpers for the matmul kernel on top of board.tcl.
#
# From a shell on the machine the board is plugged into:
#   xsdb software/jtag/matmul.tcl artifacts/matmul/<run>          ;# self-test
#   xsdb software/jtag/matmul.tcl artifacts/matmul/<run> -live    ;# type your own matrices
#
# From the xsdb prompt:
#   xsdb% source software/jtag/matmul.tcl
#   xsdb% board_open artifacts/matmul/<run>
#   xsdb% mm_check [mm_load my_a.txt] [mm_random]
#   xsdb% mm_live

source [file join [file dirname [info script]] board.tcl]

proc mm_size {} {
    lassign [board::array_info a] offset width depth
    return [expr {int(sqrt($depth))}]
}

proc mm_limits {} {
    lassign [board::array_info a] offset width depth
    return [list [expr {-(1 << ($width - 1))}] [expr {(1 << ($width - 1)) - 1}]]
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

proc mm_tokens {text} { regexp -all -inline {[^\s,;]+} $text }

# n*n integers in row-major order -> matrix; rejects values the kernel's
# input type cannot hold rather than letting them wrap silently.
proc mm_from_values {values} {
    set n [mm_size]
    lassign [mm_limits] lo hi
    if {[llength $values] != $n * $n} { error "need [expr {$n * $n}] numbers, got [llength $values]" }
    foreach v $values {
        if {![string is entier -strict $v] || $v < $lo || $v > $hi} { error "'$v' is not an integer in $lo..$hi" }
    }
    set rows {}
    for {set i 0} {$i < $n} {incr i} { lappend rows [lrange $values [expr {$i * $n}] [expr {$i * $n + $n - 1}]] }
    return $rows
}

# Text file of n*n integers separated by spaces, commas or newlines.
proc mm_load {path} {
    set f [open $path]; set text [read $f]; close $f
    return [mm_from_values [mm_tokens $text]]
}

proc mm_print {m} {
    foreach row $m {
        set cells {}
        foreach v $row { lappend cells [format %12s $v] }
        puts [join $cells ""]
    }
}

# Runs one multiplication on the FPGA, shows it, and compares with software.
# Returns 0 on a match.
proc mm_check {a b} {
    set got [mm_multiply $a $b]
    set want [mm_reference $a $b]
    puts "A:"; mm_print $a
    puts "B:"; mm_print $b
    puts "FPGA result A x B:"; mm_print $got
    set bad {}
    set n [llength $want]
    for {set i 0} {$i < $n} {incr i} {
        for {set j 0} {$j < $n} {incr j} {
            if {[lindex $got $i $j] != [lindex $want $i $j]} {
                lappend bad "c\[$i\]\[$j\]: FPGA [lindex $got $i $j], expected [lindex $want $i $j]"
            }
        }
    }
    if {![llength $bad]} {
        puts "MATCH: all [expr {$n * $n}] values equal the software reference"
        return 0
    }
    puts "MISMATCH in [llength $bad] of [expr {$n * $n}] values:"
    foreach line [lrange $bad 0 9] { puts "  $line" }
    return 1
}

# Asks for one matrix until the entry is valid. Returns a matrix, or "quit".
proc mm_prompt {in label previous} {
    while 1 {
        puts -nonewline "$label> "; flush stdout
        if {[gets $in line] < 0} { return quit }
        if {[string trim $line] eq ""} continue
        if {[catch {mm_parse_entry $in $line $previous} result]} {
            puts "  $result"
            continue
        }
        return $result
    }
}

# One entry: a command, or numbers that may continue on the following lines.
proc mm_parse_entry {in line previous} {
    set words [mm_tokens $line]
    set cmd [lindex $words 0]
    switch -- $cmd {
        q - quit     { return quit }
        i - identity { return [mm_identity] }
        z - zero     { return [mm_constant 0] }
        r - random   {
            if {[llength $words] == 3} { return [mm_random {*}[lrange $words 1 2]] }
            return [mm_random]
        }
        f - file     { return [mm_load [string trim [string range [string trim $line] [string length $cmd] end]]] }
        s - same     {
            if {$previous eq ""} { error "no previous matrix yet" }
            return $previous
        }
    }
    set need [expr {[mm_size] ** 2}]
    set values $words
    while {[llength $values] < $need} {
        foreach v $values { if {![string is entier -strict $v]} { error "'$v' is not a number or command" } }
        puts -nonewline "  [llength $values]/$need> "; flush stdout
        if {[gets $in line] < 0} { return quit }
        lappend values {*}[mm_tokens $line]
    }
    return [mm_from_values $values]
}

# Keeps the programmed board and asks for A and B repeatedly. Returns the
# number of mismatching multiplications.
proc mm_live {{in stdin}} {
    lassign [mm_limits] lo hi
    set n [mm_size]
    puts "Enter A and B as $n x $n matrices of integers in $lo..$hi."
    puts "  [expr {$n * $n}] numbers, row by row, on one or more lines (spaces or commas)"
    puts "  random \[lo hi\] | identity | zero | same | file <path> | quit"
    set a ""; set b ""; set runs 0; set failed 0
    while 1 {
        set a [mm_prompt $in A $a]
        if {$a eq "quit"} break
        set b [mm_prompt $in B $b]
        if {$b eq "quit"} break
        incr runs
        incr failed [mm_check $a $b]
        puts ""
    }
    puts "$runs multiplication(s), $failed mismatch(es)"
    return $failed
}

# Returns 0 on success so it can be an exit code.
proc mm_selftest {{random_cases 10}} {
    lassign [mm_limits] lo hi
    set cases [list \
        zero     "A = 0: every output is overwritten" [mm_constant 0]   [mm_random] \
        identity "B = I: A x I = A, no transposition" [mm_random]       [mm_identity] \
        extremes "min x max: 64-bit signed results"   [mm_constant $lo] [mm_constant $hi]]
    for {set t 0} {$t < $random_cases} {incr t} {
        lappend cases "random$t" "full-range random" [mm_random $lo $hi] [mm_random $lo $hi]
    }
    set total [expr {[llength $cases] / 4}]
    set failed 0
    foreach {name about a b} $cases {
        set got [mm_multiply $a $b]
        set want [mm_reference $a $b]
        if {$got eq $want} {
            puts [format "PASS %-9s %s" $name $about]
        } else {
            incr failed
            puts [format "FAIL %-9s %s" $name $about]
            puts "A:"; mm_print $a
            puts "B:"; mm_print $b
            puts "expected:"; mm_print $want
            puts "got:"; mm_print $got
        }
    }
    puts [expr {$failed ? "$failed of $total cases FAILED" : "PASS: all $total cases match"}]
    return [expr {$failed != 0}]
}

if {[info exists argv] && [llength $argv] > 0} {
    set live [expr {"-live" in $argv}]
    board_open {*}[lsearch -all -inline -not -exact $argv -live]
    exit [expr {$live ? [mm_live] != 0 : [mm_selftest]}]
}
