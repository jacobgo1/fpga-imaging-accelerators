#!/bin/sh
# This is a simple script used on PicoBoB to process a single capture with this kernel,
# as cnn-1d-picobob's process-capture.sh. It takes the capture name as an optional
# argument with a default value of 'hsi0'. The labels go next to the capture as
# KERNEL-labels.bin (sea 0, land 1, cloud 2), the log to its logs/.

CAPTURE_NAME="${1:-hsi0}"
PREFIX=/media/sd-pl
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
NAME="$(basename "$APP_DIR")"
DATA_PATH=$PREFIX/$CAPTURE_NAME
LOG_FILE=$DATA_PATH/logs/$NAME.log
INPUT_PATH="$DATA_PATH/binned_cube.bip"
OUTPUT_PATH="$DATA_PATH/$NAME-labels.bin"
CAPTURE_CONFIG="$DATA_PATH/capture_config.ini"

mkdir -p "$DATA_PATH/logs"
"$APP_DIR/classify" --input "$INPUT_PATH" --output "$OUTPUT_PATH" \
    --capture-config "$CAPTURE_CONFIG" --log-level debug --log-file "$LOG_FILE"
