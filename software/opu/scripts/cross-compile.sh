#!/bin/bash

# Cross compile CNN application

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

mkdir -p $SCRIPTPATH/../build-cross 
build_cmd="cd build-cross && cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-arm.cmake .. && make"
$SCRIPTPATH/start_docker $build_cmd
