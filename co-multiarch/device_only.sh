#!/bin/bash
set -x

compile_device_cc() {
    local input_file="$1"
    local object_file="$2"
    local arch="$3"
    local bc_file="$object_file.bc"

    # Two stage compile: C++ to BC, BC to ELF
    # This is necessary because certain variants of clang cannot do
    # device only compilation directly from the frontend (i.e. if you
    # pass a -triple for a GPU, it still is hardcoded to only do host
    # compilation). Instead of fighting it, use the frontend to emit
    # bitcode explicitly (default on Windows, forced on others) with
    # `-emit-llvm`.
    # Then use the cc1 backend to compile it to elf.
    # Finally link all device elf files into a combined hsaco file that
    # can be loaded onto the device directly as a module.
    #
    # There are additional things that can be done to create fat multi-arch
    # archives, etc.
    # This procedure should work on all platforms. It can be verified with:
    #   llvm-readelf -a merged_gfx1151.hsaco | grep -E 'amdhsa|Machine|Type'
    # This should show kernels, and the correct gfx version.
    clang++ -x hip -fgpu-rdc -O3 -std=c++17 \
        --offload-arch=$arch \
        --offload-device-only \
        -emit-llvm \
        -c "$input_file" -o "$bc_file"

    clang++ -cc1 \
        -triple amdgcn-amd-amdhsa \
        -target-cpu $arch \
        -fcuda-is-device \
        -fgpu-rdc -O3 \
        -emit-obj \
        -o "$object_file" \
        -x ir "$bc_file"
}

rm -f *.o *.bc *.hsaco

archs="gfx1100 gfx1151 gfx942 gfx906 gfx1101 gfx1201 gfx1200 gfx1010 gfx906 gfx1030 gfx900"
# Compile in parallel for all architectures.
for arch in $archs; do
    echo ":: Compiling arch $arch"
    compile_device_cc device_fragment_1.cc device_fragment_1_$arch.o $arch &
    compile_device_cc device_fragment_2.cc device_fragment_2_$arch.o $arch &
done
wait

# Link in parallel for each architecture.
for arch in $archs; do
    ld.lld -shared device_fragment_1_$arch.o device_fragment_2_$arch.o \
        -o merged_$arch.hsaco &
done
wait