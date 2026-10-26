
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

extern "C" void run_normalize(
    float* h_data, float* h_means, float* h_stds, int N, int F
) {
    float *d_data, *d_means, *d_stds;
    cudaMalloc(&d_data,  sizeof(float) * N * F);
    cudaMalloc(&d_means, sizeof(float) * F);
    cudaMalloc(&d_stds,  sizeof(float) * F);
    cudaMemcpy(d_data,  h_data,  sizeof(float) * N * F, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, h_means, sizeof(float) * F,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_stds,  h_stds,  sizeof(float) * F,     cudaMemcpyHostToDevice);
    dim3 block(32, 32);
    dim3 grid((F + 31) / 32, (N + 31) / 32);
    normalize_kernel<<<grid, block>>>(d_data, d_means, d_stds, N, F);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, sizeof(float) * N * F, cudaMemcpyDeviceToHost);
    cudaFree(d_data); cudaFree(d_means); cudaFree(d_stds);
}
