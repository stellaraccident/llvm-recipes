// device_fragment_2.cc

#include <hip/hip_runtime.h>

extern "C" __global__ void vector_scale(float* data, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= scale;
    }
}
