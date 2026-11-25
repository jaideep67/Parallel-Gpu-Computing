# Data Description: CUDA Warp Divergence Benchmarking

## Dataset Characteristics
The benchmark operates on a synthetic floating-point array of size:
$$N = 2^{24} = 16,777,216 \text{ single-precision elements (FP32)}$$
Total buffer size in GPU global memory:
$$16,777,216 \times 4 \text{ bytes} = 67,108,864 \text{ bytes} = 64 \text{ MiB}$$

## Data Generation Scheme
To guarantee deterministic execution and evaluate divergence accurately across threads without memory read bias, input data is procedurally generated using a linear congruential pseudorandom pattern:

```cpp
for (int i = 0; i < N; ++i) {
    unsigned int x = static_cast<unsigned int>(i);
    x = x * 1664525u + 1013904223u;
    h_input[i] = static_cast<float>(x & 0xFFFFu) / 65535.0f;
}
```

- **Value Range**: Normalized floating-point values in $[0.0, 1.0]$.
- **Branch Threshold**: $0.5$.
- **Distribution**: Uniformly distributed around 0.5 so that approximately 50% of the elements trigger path A (`workA`), and 50% trigger path B (`workB`), producing maximal intra-warp branch divergence in the data-dependent kernel.
