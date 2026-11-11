FROM ubuntu:20.04

# Fix timezone issue
ENV TZ=Europe/Oslo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install general tools and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    sudo

# Install cross-compilation tools for ARM
RUN apt-get update && apt-get install -y \
    binutils-aarch64-linux-gnu \
    binutils-arm-linux-gnueabihf \
    g++-aarch64-linux-gnu \
    g++-arm-linux-gnueabihf \
    gcc-aarch64-linux-gnu \
    gcc-arm-linux-gnueabihf

# Make a HYPSO user
RUN adduser --disabled-password --gecos '' hypso && \
    usermod -aG sudo hypso && \
    echo "hypso ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER hypso
WORKDIR /home/hypso/
