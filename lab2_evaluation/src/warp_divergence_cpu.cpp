// CPU Baseline implementation for Warp Divergence comparison
// Simulates the equivalent operations of warp_divergence.cu on host CPU
//
// Compile with:
//   g++ -O2 -std=c++14 warp_divergence_cpu.cpp -o warp_divergence_cpu
//   (or on Windows: cl /O2 /EHsc /std:c++14 warp_divergence_cpu.cpp)

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

constexpr int N = 1 << 24; // 16,777,216 elements
constexpr int REPEATS = 20;
constexpr int WARP_SIZE = 32;

inline float workA(float x) {
    for (int k = 0; k < 8; ++k)
        x = x * 1.0001f + 0.25f;
    return x;
}

inline float workB(float x) {
    for (int k = 0; k < 8; ++k)
        x = x * 0.9999f - 0.25f;
    return x;
}

void cpuDivergent(const float* input, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        if ((i % 2) == 0)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

void cpuAligned(const float* input, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        if (((i / WARP_SIZE) % 2) == 0)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

void cpuDataDependent(const float* input, float* output, int n, float threshold) {
    for (int i = 0; i < n; ++i) {
        if (input[i] > threshold)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

void cpuBranchFree(const float* input, float* output, int n, float threshold) {
    for (int i = 0; i < n; ++i) {
        float a = workA(input[i]);
        float b = workB(input[i]);
        output[i] = (input[i] > threshold) ? a : b;
    }
}

int main() {
    std::cout << "CPU Warp Divergence Baseline Benchmark" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "Array elements: " << N << std::endl;
    std::cout << "Kernel repetitions: " << REPEATS << std::endl << std::endl;

    std::vector<float> input(N);
    std::vector<float> output(N);

    for (int i = 0; i < N; ++i) {
        unsigned int x = static_cast<unsigned int>(i);
        x = x * 1664525u + 1013904223u;
        input[i] = static_cast<float>(x & 0xFFFFu) / 65535.0f;
    }

    const float threshold = 0.5f;

    auto benchmark = [&](const std::string& name, auto func) {
        // Warm up
        func();

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPEATS; ++r) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
        double avgMs = totalMs / REPEATS;

        std::cout << std::left << std::setw(30) << name
                  << std::right << std::fixed << std::setprecision(4)
                  << avgMs << " ms" << std::endl;
        return avgMs;
    };

    std::cout << "Average CPU execution time (" << REPEATS << " repetitions):" << std::endl;
    benchmark("CPU Alternating Branch", [&]() { cpuDivergent(input.data(), output.data(), N); });
    benchmark("CPU Chunk-aligned Branch", [&]() { cpuAligned(input.data(), output.data(), N); });
    benchmark("CPU Data-dependent Branch", [&]() { cpuDataDependent(input.data(), output.data(), N, threshold); });
    benchmark("CPU Branch-free (both paths)", [&]() { cpuBranchFree(input.data(), output.data(), N, threshold); });

    return 0;
}
