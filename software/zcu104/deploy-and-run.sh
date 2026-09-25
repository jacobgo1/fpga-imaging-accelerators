#!/usr/bin/env bash
# Sync this repo to a remote host over ssh and run a main.py stage there --
# the board's own native g++/python3 for the A53 track (deploy-and-run.sh
# ubuntu@zcu104.local), or a build server's Vitis HLS/Vivado for csynth/
# synth/impl/bitstream. No GUI, no manual file copying: rsync out, ssh in,
# rsync reports/ and artifacts/ back.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: deploy-and-run.sh <user@host> [-- <main.py args...>]

Defaults to: native --kernel justoliunet
JNET_BENCH_REPS, if set in this shell, is forwarded to the board.
REMOTE_SETUP, if set, is run before main.py on the remote host -- use it to
source Vitis/Vivado's settings64.sh on a build server (same requirement as
the main README's "Load AMD tools": it has to happen in the same shell
that then calls main.py).

Examples:
  deploy-and-run.sh ubuntu@zcu104.local
  deploy-and-run.sh ubuntu@192.168.1.50 -- native --kernel justoliunet
  JNET_BENCH_REPS=2000 deploy-and-run.sh ubuntu@zcu104.local

  REMOTE_SETUP='source /tools/Xilinx/Vivado/2024.2/settings64.sh && source /tools/Xilinx/Vitis/2024.2/settings64.sh' \
      deploy-and-run.sh you@build-server -- bitstream --kernel justoliunet --config config/project.local.json
EOF
}

target="${1:-}"
case "$target" in
    ""|-h|--help) usage; exit $([ -z "$target" ] && echo 2 || echo 0) ;;
esac
shift || true
[ "${1:-}" = "--" ] && shift
args=("$@")
[ ${#args[@]} -gt 0 ] || args=(native --kernel justoliunet)

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_dir="fpga-imaging-accelerators"

echo "== Syncing repo to $target:~/$remote_dir =="
rsync -az --delete \
    --exclude '.git' --exclude 'build' --exclude 'reports' --exclude 'artifacts' \
    --exclude '__pycache__' \
    "$root/" "$target:~/$remote_dir/"

env_prefix=""
[ -n "${JNET_BENCH_REPS:-}" ] && env_prefix="JNET_BENCH_REPS=$JNET_BENCH_REPS "
setup_prefix=""
[ -n "${REMOTE_SETUP:-}" ] && setup_prefix="${REMOTE_SETUP} && "

echo "== Running on $target: python3 main.py ${args[*]} =="
# shellcheck disable=SC2029
ssh "$target" "cd $remote_dir && ${setup_prefix}${env_prefix}python3 main.py ${args[*]}"

echo "== Pulling reports and artifacts back =="
rsync -az "$target:~/$remote_dir/reports/" "$root/reports/"
rsync -az "$target:~/$remote_dir/artifacts/" "$root/artifacts/" \
    || echo "(no artifacts/ on remote -- normal for a native/csim/csynth-only run)"
echo "Done. See reports/<kernel>/latest for logs, artifacts/<kernel>/latest for any bitstream/IP/checkpoints."
