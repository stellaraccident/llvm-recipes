#!/bin/bash
set -x

for arch in gfx1100 gfx1151; do
    echo ":: Compiling arch $arch"
    clang++ -x hip -fgpu-rdc -O3 -std=c++17 \
        --offload-arch=$arch \
        --offload-device-only \
        -c device_fragment_1.cc -o device_fragment_1_$arch.o
    clang++ -x hip -fgpu-rdc -O3 -std=c++17 \
        --offload-arch=$arch \
        --offload-device-only \
        -c device_fragment_2.cc -o device_fragment_2_$arch.o

    ld.lld -shared device_fragment_1_$arch.o device_fragment_2_$arch.o \
        -o merged_$arch.hsaco
done
