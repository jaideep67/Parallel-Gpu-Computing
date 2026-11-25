// CUDA Warp Divergence Lab Evaluation 2
// Implements:
//   1. Divergent branch kernel
//   2. Warp-aligned branch kernel
//   3. Data-dependent branch kernel
//   4. Branch-free (predicated/select-style) version
//
// Compile on Ubuntu/Linux with:
//   nvcc -O2 -std=c++14 warp_divergence.cu -o warp_divergence
//
// Run:
//   ./warp_divergence
//
// Requirements:
//   - NVIDIA GPU
//   - NVIDIA driver
//   - CUDA Toolkit (nvcc)
//
// Notes:
//   - NVIDIA CUDA warp size is 32 on the target architectures used here.
//   - Timings are measured with CUDA Events and are device-kernel timings.
//   - Run more than once and compare averages; results depend on GPU/load/compiler.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <string>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " -> " << cudaGetErrorString(err) << std::endl;     \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

constexpr int WARP_SIZE = 32;
constexpr int N = 1 << 24;       // 16,777,216 elements
constexpr int THREADS = 256;     // 8 warps per block
constexpr int REPEATS = 20;

// A small amount of work makes the branch behavior easier to measure.
// The result depends on input, so the compiler cannot remove the calculation.
__device__ __forceinline__ float workA(float x)
{
    #pragma unroll 1
    for (int k = 0; k < 8; ++k)
        x = x * 1.0001f + 0.25f;
    return x;
}

__device__ __forceinline__ float workB(float x)
{
    #pragma unroll 1
    for (int k = 0; k < 8; ++k)
        x = x * 0.9999f - 0.25f;
    return x;
}

// CP2 - Kernel 1: adjacent threads in a warp alternate between A and B.
// For each warp, approximately half the lanes take each path.
__global__ void divergentKernel(const float* input, float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if ((i % 2) == 0)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

// CP2 - Kernel 2: the branch condition changes only at 32-thread boundaries.
// All lanes in a warp take the same path (assuming aligned warp indexing).
__global__ void alignedKernel(const float* input, float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (((i / WARP_SIZE) % 2) == 0)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

// CP3 - Data-dependent branch: each thread chooses a path from its input value.
__global__ void dataDependentKernel(const float* input, float* output,
                                    int n, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (input[i] > threshold)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

// CP3 - Branch-free alternative: calculate both candidate values and select.
// This avoids control-flow divergence, but does extra arithmetic.
// It is not guaranteed to be faster; measure it on the actual GPU.
__global__ void branchFreeKernel(const float* input, float* output,
                                 int n, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = workA(input[i]);
        float b = workB(input[i]);
        output[i] = (input[i] > threshold) ? a : b;
    }
}

float measureKernel(const std::string& name,
                    const float* d_input, float* d_output, int n,
                    int blocks, float threshold,
                    int kernelType)
{
    // Warm-up launch (not included in timing).
    switch (kernelType) {
        case 0: divergentKernel<<<blocks, THREADS>>>(d_input, d_output, n); break;
        case 1: alignedKernel<<<blocks, THREADS>>>(d_input, d_output, n); break;
        case 2: dataDependentKernel<<<blocks, THREADS>>>(d_input, d_output, n, threshold); break;
        case 3: branchFreeKernel<<<blocks, THREADS>>>(d_input, d_output, n, threshold); break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPEATS; ++r) {
        switch (kernelType) {
            case 0: divergentKernel<<<blocks, THREADS>>>(d_input, d_output, n); break;
            case 1: alignedKernel<<<blocks, THREADS>>>(d_input, d_output, n); break;
            case 2: dataDependentKernel<<<blocks, THREADS>>>(d_input, d_output, n, threshold); break;
            case 3: branchFreeKernel<<<blocks, THREADS>>>(d_input, d_output, n, threshold); break;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float averageMs = totalMs / REPEATS;
    std::cout << std::left << std::setw(30) << name
              << std::right << std::fixed << std::setprecision(4)
              << averageMs << " ms" << std::endl;
    return averageMs;
}

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable NVIDIA GPU was detected." << std::endl;
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "CUDA Warp Divergence Practical" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Warp size reported by device: " << prop.warpSize << std::endl;
    std::cout << "Array elements: " << N << std::endl;
    std::cout << "Threads per block: " << THREADS << std::endl;
    std::cout << "Kernel repetitions: " << REPEATS << std::endl << std::endl;

    if (prop.warpSize != WARP_SIZE) {
        std::cout << "Warning: this example's warp-alignment logic assumes "
                  << WARP_SIZE << "-thread NVIDIA warps." << std::endl;
    }

    std::vector<float> h_input(N);
    // Deterministic mixed data: values are distributed around threshold 0.5.
    // The pattern produces both outcomes within many warps.
    for (int i = 0; i < N; ++i) {
        unsigned int x = static_cast<unsigned int>(i);
        x = x * 1664525u + 1013904223u;
        h_input[i] = static_cast<float>(x & 0xFFFFu) / 65535.0f;
    }

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    int blocks = (N + THREADS - 1) / THREADS;
    const float threshold = 0.5f;

    std::cout << "Average kernel execution time (" << REPEATS
              << " repetitions):" << std::endl;

    float tDivergent = measureKernel("Divergent branch",
                                     d_input, d_output, N, blocks, threshold, 0);
    float tAligned = measureKernel("Warp-aligned branch",
                                   d_input, d_output, N, blocks, threshold, 1);
    float tData = measureKernel("Data-dependent branch",
                                d_input, d_output, N, blocks, threshold, 2);
    float tBranchFree = measureKernel("Branch-free selection",
                                      d_input, d_output, N, blocks, threshold, 3);

    std::cout << std::endl << "Comparison:" << std::endl;
    if (tAligned > 0.0f)
        std::cout << "Divergent / aligned time ratio: "
                  << std::setprecision(3) << (tDivergent / tAligned) << "x"
                  << std::endl;
    if (tBranchFree > 0.0f)
        std::cout << "Data-dependent / branch-free time ratio: "
                  << std::setprecision(3) << (tData / tBranchFree) << "x"
                  << std::endl;

    std::cout << std::endl
              << "Note: ratios above 1 mean the first kernel took longer in this run."
              << std::endl
              << "Do not assume a particular winner; GPU model, compiler, clocks,"
              << std::endl
              << "memory behavior, and workload affect the measurements."
              << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
