#*****************************************************************************************
# An HLS kernel in the OPU system's design, for opu-system's vivado/run_tcl_script.sh.
# scripts/opu.sh fills in the @...@ words and puts this in opu-system's
# vivado/tcl-scripts/tmp/, as cnn-1d-picobob's build-firmware.sh does with its own.
#*****************************************************************************************

#*****************************************************************************************
# Setting paths (run_tcl_script.sh switches these two for Vivado without Docker)
#*****************************************************************************************

# Use this if you are not using docker:
#set repoDir /home/hypso/opu-system

# Use this if you are using docker:
set repoDir /home/hypso

set projectDir $repoDir/vivado/projects

# Name of the project
set projectName @PROJECT@

# The HLS kernel: its top function and the clock period HLS built it for (ns)
set hlsTop @TOP@
set hlsClockNs @CLOCK_NS@

#*****************************************************************************************
# Run tcl scripts to make the project
#*****************************************************************************************

# Check that vivado is using the correct version
source $repoDir/vivado/tcl-scripts/check_vivado.tcl

# Set up processing system block for the board
source $repoDir/vivado/tcl-scripts/@BOARD_SCRIPT@

# Create base project for opu
source $repoDir/vivado/tcl-scripts/opu_base.tcl

# Add the HLS kernel
source $repoDir/vivado/tcl-scripts/tmp/add_hls_kernel.tcl

#*****************************************************************************************
# Save project and generate bitstream
#*****************************************************************************************
source $repoDir/vivado/tcl-scripts/finish_design.tcl
puts "The script finished"
