#!/bin/bash
# get-petalinux.sh [HOST]: copy the PetaLinux Docker image opu-system builds with
# (petalinux:2023.1) from a machine that has it, over ssh, so this machine can run
# scripts/opu.sh. HOST is an ssh host, default fpgaws (the FPGA workstation). This saves
# downloading the installer with an AMD account and building the image
# (opu-system's docker/setup-petalinux-docker.sh), which is the other way.
#
# About 17 GB, compressed on the way. Also lets Docker read opu-system's petalinux folder on
# SELinux machines (Fedora), which run-petalinux-docker.sh mounts without :z.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -le 1 ] || die "usage: scripts/get-petalinux.sh [HOST]"
host="${1:-fpgaws}"
image=petalinux:2023.1
need docker "Install Docker first."
need zstd "Install zstd first (dnf install zstd)."

if docker image inspect "$image" >/dev/null 2>&1; then
    echo "$image is already here"
else
    # The layers' digests, not the image ID: Docker's containerd store gives the same image
    # another ID.
    layers='{{range .RootFS.Layers}}{{.}} {{end}}'
    remote_layers="$(ssh "$host" "docker image inspect $image --format '$layers'")" \
        || die "no $image on $host"
    echo "Copying $image from $host (about 17 GB, this takes a while)"
    # The remote side compresses only if it has zstd; docker load takes either.
    ssh "$host" "if command -v zstd >/dev/null; then docker save $image | zstd -3 -T0 -c; else docker save $image; fi" \
        | zstd -d -c -f | docker load
    [ "$(docker image inspect "$image" --format "$layers")" = "$remote_layers" ] \
        || die "$image here is not the one on $host"
    echo "$image copied, the same as on $host"
fi

OPU="${FPGA_OPU_SYSTEM:-$FPGA_ROOT/../opu-system}"
if command -v getenforce >/dev/null && [ "$(getenforce)" = Enforcing ] && [ -d "$OPU/petalinux" ]; then
    # Only the SELinux label changes, nothing git sees.
    chcon -R -t container_file_t "$OPU/petalinux"
    echo "Docker may now read $OPU/petalinux (SELinux)"
fi
