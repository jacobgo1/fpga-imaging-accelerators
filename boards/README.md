# Board integration contract

To build a system bitstream, write `boards/<name>/system.tcl` and point
`board_script` in `config/project.json` at it. Start from a working Vivado block
design exported with `write_bd_tcl`, then adapt it to this contract. Track the
Tcl and XDC, not `.xpr` or generated block-design files.

`scripts/vivado.tcl` has already created a project with the configured part and
added the exported HLS IP to its catalog before sourcing your script. Populate
**that** project; do not launch synthesis or implementation yourself.

| Variable | Meaning |
| --- | --- |
| `cfg(root)` | Absolute repository path |
| `cfg(run_dir)` | This run's build directory — keep generated files here |
| `cfg(part)` / `cfg(clock_ns)` | Device and HLS clock target |
| `cfg(top)` | HLS kernel top name (not the board-level top) |
| `ip_repo` | Exported HLS IP catalog directory |
| `rtl_dir` | HLS synthesized Verilog directory |

Your script should:

1. Set `board_part` if you use board presets.
2. Add tracked RTL and XDC with `file join $cfg(root) ...`.
3. Instantiate the HLS IP with the processor, memory, interconnect and
   clock/reset logic it needs; assign addresses; validate, save and generate the
   block design; add its HDL wrapper.
4. Set the `sources_1` fileset top to the real board-level top.
5. Constrain clocks, pins and I/O standards, and make sure the actual kernel
   clock matches the HLS target.

Never suppress unconstrained-pin DRCs to force a bitstream, and read the
unconstrained-path section of the timing report — the automatic negative-slack
check cannot detect constraints you forgot to write.

`synth` then synthesizes the whole system, `impl` routes it, and `bitstream`
writes the `.bit` (plus debug probes, and an `.xsa` when `export_xsa` is true).
Programming the board and building boot images are separate from this flow.
