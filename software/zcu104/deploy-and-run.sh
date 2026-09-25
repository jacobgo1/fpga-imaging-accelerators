#!/usr/bin/env bash
# Sync this repo to the ZCU104 over ssh and run a main.py stage there --
# the A53 equivalent of running `python3 main.py <stage> --kernel <kernel>`
# locally, just executed by the board's own native g++/python3 instead of
# yours. No GUI, no manual file copying: rsync out, ssh in, rsync the
# reports back.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: deploy-and-run.sh <user@host> [-- <main.py args...>]

Defaults to: native --kernel justoliunet
JNET_BENCH_REPS, if set in this shell, is forwarded to the board.

Examples:
  deploy-and-run.sh ubuntu@zcu104.local
  deploy-and-run.sh ubuntu@192.168.1.50 -- native --kernel justoliunet
  JNET_BENCH_REPS=2000 deploy-and-run.sh ubuntu@zcu104.local
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

echo "== Running on $target: python3 main.py ${args[*]} =="
# shellcheck disable=SC2029
ssh "$target" "cd $remote_dir && ${env_prefix}python3 main.py ${args[*]}"

echo "== Pulling reports back =="
rsync -az "$target:~/$remote_dir/reports/" "$root/reports/"
echo "Done. See reports/<kernel>/latest locally -- now including this board run's manifest.json."
