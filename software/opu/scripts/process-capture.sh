# This is a simple script used on PicoBoB to process a single capture. This
# script takes the capture name as an optional argument with a default value of
# 'hsi0'.

CAPTURE_NAME="${1:-hsi0}"
PREFIX=/media/sd-pl
# The kernel's name: the folder this script is in, /media/sd-pl/KERNEL
NAME=$(basename $(dirname $(readlink -f $0)))
CNN_PATH=$PREFIX/$NAME/cnn_accelerator
DATA_PATH=$PREFIX/$CAPTURE_NAME
LOG_FILE=$DATA_PATH/logs/$NAME.log
INPUT_PATH="$DATA_PATH/binned_cube.bip"
OUTPUT_PATH="$DATA_PATH/$NAME-labels.bin"
CAPTURE_CONFIG="$DATA_PATH/capture_config.ini"

$CNN_PATH --input $INPUT_PATH --output $OUTPUT_PATH \
    --capture-config $CAPTURE_CONFIG --log-level debug --log-file $LOG_FILE
