#!/bin/sh
# HYPSO-2: classify one capture on the OPU, as driver_ntnu's script does for
# the 1D CNN:  process-capture.sh [CAPTURE]  (default hsi0)
# The labels are in the HYPSO tools' order (0 sea, 1 land, 2 cloud).
CAPTURE_NAME="${1:-hsi0}"
PREFIX=/media/sd-pl
DRIVER=$PREFIX/jopt_driver_zynq7030
DATA=$PREFIX/$CAPTURE_NAME

mkdir -p "$DATA/logs"
"$DRIVER/jopt_driver" --header "$DRIVER/xjustounetsimple_opt_zynq7030_hw.h" \
    --input "$DATA/binned_cube.bip" --capture-config "$DATA/capture_config.ini" \
    --labels "$DATA/jopt-labels.bin" --hypso-labels \
    > "$DATA/logs/jopt-driver.log" 2>&1
