# HLS directives for justounetsimple_opt_zynq7020 (main.py sources config/KERNEL.tcl
# after create_clock). The Zynq-7020 has 220 DSPs and conv1 takes 216 of them:
# conv3x3_packed binds its multiplies to DSPs itself. Every other multiply goes
# into LUTs: conv2-4's, the z-score's, and the address and patch-count
# arithmetic, which HLS would otherwise put on the last few DSPs and beyond.
config_op mul -impl fabric
