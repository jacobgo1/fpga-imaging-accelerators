#!/bin/bash
# kernels.sh: every kernel, and the boards there are.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
list_kernels
echo "boards: $(list_boards | tr '\n' ' ')" >&2
