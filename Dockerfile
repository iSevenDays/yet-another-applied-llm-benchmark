# Use an official Ubuntu as a parent image
FROM ubuntu:22.04

# Set ARG for TARGETPLATFORM
ARG TARGETPLATFORM

# Update the system and install Python and build dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    gcc \
    curl \
    sqlite3 \
    gdb \
    libssl-dev \
    wget \
    clang \
    libxml2-dev \
    libncurses5-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND noninteractive

# Add the deadsnakes PPA, which contains newer Python versions
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.12
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git

# Remove the EXTERNALLY-MANAGED file to allow pip installations
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# Download and install pip using the get-pip.py script
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Install required Python packages using pip
RUN pip install numpy scipy numba Pillow jax jaxlib python-chess torch

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Ensure Rust binaries are in PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Swift based on the target platform
RUN set -ex && \
    SWIFT_PLATFORM=ubuntu22.04 && \
    SWIFT_BRANCH=swift-5.9.2-release && \
    SWIFT_VERSION=swift-5.9.2-RELEASE && \
    SWIFT_WEBROOT=https://download.swift.org && \
    case "$TARGETPLATFORM" in \
        linux/amd64) \
            SWIFT_WEBDIR="$SWIFT_WEBROOT/$SWIFT_BRANCH/$(echo $SWIFT_PLATFORM | tr -d .)/$SWIFT_VERSION" && \
            SWIFT_BIN_URL="$SWIFT_WEBDIR/$SWIFT_VERSION-$SWIFT_PLATFORM.tar.gz" \
            ;; \
        linux/arm64*) \
            SWIFT_WEBDIR="$SWIFT_WEBROOT/$SWIFT_BRANCH/$(echo $SWIFT_PLATFORM | tr -d .)-aarch64/$SWIFT_VERSION" && \
            SWIFT_BIN_URL="$SWIFT_WEBDIR/$SWIFT_VERSION-$SWIFT_PLATFORM-aarch64.tar.gz" \
            ;; \
        *) \
            echo "Unsupported architecture: $TARGETPLATFORM" && exit 1 \
            ;; \
    esac && \
    wget -q $SWIFT_BIN_URL && \
    tar xzf ${SWIFT_VERSION}-${SWIFT_PLATFORM}*.tar.gz --directory / --strip-components=1 && \
    rm ${SWIFT_VERSION}-${SWIFT_PLATFORM}*.tar.gz && \
    swift --version

# Add Swift to PATH
ENV PATH="/usr/share/swift/usr/bin:${PATH}"

# Create a symlink for python3
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Set the working directory in the container
WORKDIR /usr/src/app

# Command to run when the container launches
CMD ["/bin/bash"]