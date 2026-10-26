
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void normalize_kernel(
    float* data, float* means, float* stds, int N, int F
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    int row     = blockIdx.y * blockDim.y + threadIdx.y;
    if (feature >= F || row >= N) return;
    int idx = row * F + feature;
    data[idx] = (data[idx] - means[feature]) / stds[feature];
}

extern "C" void run_normalize_streams(
    float* h_data, float* h_means, float* h_stds, int N, int F
) {
    int half = N / 2;
    float *d_data0, *d_data1, *d_means, *d_stds;
    cudaMalloc(&d_data0, sizeof(float) * half * F);
    cudaMalloc(&d_data1, sizeof(float) * half * F);
    cudaMalloc(&d_means, sizeof(float) * F);
    cudaMalloc(&d_stds,  sizeof(float) * F);
    cudaMemcpy(d_means, h_means, sizeof(float) * F, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stds,  h_stds,  sizeof(float) * F, cudaMemcpyHostToDevice);
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    dim3 block(32, 32);
    dim3 grid((F + 31) / 32, (half + 31) / 32);
    cudaMemcpyAsync(d_data0, h_data, sizeof(float)*half*F, cudaMemcpyHostToDevice, stream0);
    normalize_kernel<<<grid, block, 0, stream0>>>(d_data0, d_means, d_stds, half, F);
    cudaMemcpyAsync(h_data, d_data0, sizeof(float)*half*F, cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(d_data1, h_data+half*F, sizeof(float)*half*F, cudaMemcpyHostToDevice, stream1);
    normalize_kernel<<<grid, block, 0, stream1>>>(d_data1, d_means, d_stds, half, F);
    cudaMemcpyAsync(h_data+half*F, d_data1, sizeof(float)*half*F, cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamDestroy(stream0); cudaStreamDestroy(stream1);
    cudaFree(d_data0); cudaFree(d_data1); cudaFree(d_means); cudaFree(d_stds);
}
