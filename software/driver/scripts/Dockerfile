# Cross compilers for both boards' ARMs, for scripts/cross-compile.sh.
FROM ubuntu:22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake make file \
        g++-arm-linux-gnueabihf \
        g++-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*
