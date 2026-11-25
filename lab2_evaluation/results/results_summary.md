# Results Summary: Warp Divergence Benchmarking

## Experimental Hardware & Environment
- **GPU**: NVIDIA RTX 4500 Ada Generation
- **Architecture**: Ada Lovelace (SM 8.9)
- **CUDA Driver & Compiler**: CUDA 13.4 (V13.4.92) / MSVC 19.51
- **Host System**: Windows 11 / x64 Native Tools
- **Test Array Size**: $N = 2^{24} = 16,777,216$ FP32 elements
- **Block Size**: 256 threads (8 warps per thread block)
- **Grid Size**: 65,536 thread blocks ($16,777,216 / 256$)
- **Repetitions**: 20 iterations timed via CUDA Events

---

## Measured Performance Data

| Kernel Variant | Branching Mechanism | Avg Time (ms) | Throughput (M elem/s) | Branch Efficiency (est.) | Active Mask Pattern |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Divergent Branch** | `(i % 2) == 0` | **0.3738 ms** | 44,882.87 | ~50% | Alternating 0x55555555 / 0xAAAAAAAA |
| **Warp-Aligned Branch** | `((i / 32) % 2) == 0` | **0.3955 ms** | 42,420.27 | 100% | Full warp active: 0xFFFFFFFF |
| **Data-Dependent Branch** | `input[i] > threshold` | **0.3648 ms** | 45,990.18 | ~50% (random) | Non-uniform per-warp bitmask |
| **Branch-Free Selection** | `(input[i] > threshold) ? a : b` | **0.3952 ms** | 42,452.47 | 100% (predicated) | Full warp active, dual computation |

---

## Key Ratios & Observations
- **Divergent / Aligned Ratio**: `0.945x`
- **Data-Dependent / Branch-Free Ratio**: `0.923x`

### Architectural Analysis (Why did branch-free take longer?)
1. **Arithmetic Overhead**: In the branch-free kernel, both `workA()` (8 multiply-adds) and `workB()` (8 multiply-adds) are calculated by **every thread**, requiring 16 arithmetic operations per thread, followed by a conditional move (`selp`). In contrast, the branching kernels only execute 8 arithmetic operations per active thread lane.
2. **Compiler Predication & SM 8.9 Independent Thread Scheduling**: On NVIDIA Ada Lovelace architecture (Compute Capability 8.9), the warp scheduler and instruction cache handle short divergent branches using hardware predication without thread serialization stalls. Since the branch bodies are small (8 loop iterations), the branch overhead is negligible compared to doubling the compute payload.
