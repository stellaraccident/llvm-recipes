#include <hip/hip_runtime.h>

#include <iostream>

// Functions marked with __device__ are executed on the device and called from the device only.
__device__ unsigned int get_thread_idx2()
{
    // Built-in threadIdx returns the 3D coordinate of the active work item in the block of threads.
    return threadIdx.x;
}

// // Functions marked with __host__ are executed on the host and called from the host.
// __host__ void print_hello_host2()
// {
//     std::cout << "Hello world from host!" << std::endl;
// }

// Functions marked with __device__ and __host__ are compiled both for host and device.
// These functions cannot use coordinate built-ins.
__device__ void print_hello2()
{
    // Only printf is supported for printing from device code.
    printf("Hello world from device or host!\n");
}

// Functions marked with __global__ are executed on the device and called from the host only.
__device__ void helloworld_kernel2()
{
    unsigned int thread_idx = get_thread_idx2();
    // Built-in blockIdx returns the 3D coorindate of the active work item in the grid of blocks.
    unsigned int block_idx = blockIdx.x;

    print_hello2();

    // Only printf is supported for printing from device code.
    printf("Hello world from device kernel block %u thread %u!\n", block_idx, thread_idx);
}
