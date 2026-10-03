```python
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 1 - MATRIX MULTIPLICATION PERFORMANCE
# ============================================================

# Implementations
implementations = [
    "Sequential",
    "OpenMP",
    "MPI",
    "CUDA"
]

# Execution time in seconds
execution_time = [
    266.477234,
    41.021555,
    226.167575,
    0.077087
]

# Baseline
sequential_time = 266.477234


# ============================================================
# 1. EXECUTION TIME GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    implementations,
    execution_time
)

plt.xlabel("Implementation")
plt.ylabel("Execution Time (seconds)")
plt.title("Matrix Multiplication - Execution Time Comparison")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "../results/execution_time_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. SPEEDUP CALCULATION
# ============================================================

speedup = [
    sequential_time / time
    for time in execution_time
]

print("\nSpeedup Results:")

for name, value in zip(implementations, speedup):
    print(
        f"{name}: {value:.2f}x"
    )


# ============================================================
# SPEEDUP GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    implementations,
    speedup
)

plt.xlabel("Implementation")
plt.ylabel("Speedup (×)")
plt.title("Matrix Multiplication - Speedup Comparison")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "../results/speedup_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. PERFORMANCE IMPROVEMENT
# ============================================================

improvement = [
    ((sequential_time - time) / sequential_time) * 100
    for time in execution_time
]

print("\nPerformance Improvement:")

for name, value in zip(implementations, improvement):
    print(
        f"{name}: {value:.2f}%"
    )


# ============================================================
# PERFORMANCE IMPROVEMENT GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    implementations,
    improvement
)

plt.xlabel("Implementation")
plt.ylabel("Performance Improvement (%)")
plt.title("Matrix Multiplication - Performance Improvement")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "../results/performance_improvement.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. CUDA TOTAL VS KERNEL TIME
# ============================================================

cuda_total = 0.077087
cuda_kernel = 0.064499

cuda_times = [
    cuda_total,
    cuda_kernel
]

cuda_labels = [
    "CUDA Total",
    "CUDA Kernel"
]

plt.figure(figsize=(8, 5))

plt.bar(
    cuda_labels,
    cuda_times
)

plt.xlabel("CUDA Execution Type")
plt.ylabel("Time (seconds)")
plt.title("CUDA Total Time vs Kernel Time")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "../results/cuda_time_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========================================")
print("Experiment 1 Graph Generation Complete")
print("========================================")

print("\nExecution Times:")

for name, time in zip(implementations, execution_time):
    print(f"{name}: {time:.6f} seconds")

print("\nSpeedup:")

for name, value in zip(implementations, speedup):
    print(f"{name}: {value:.2f}x")

print("\nPerformance Improvement:")

for name, value in zip(implementations, improvement):
    print(f"{name}: {value:.2f}%")

print("\nGraphs saved in:")
print("../results/")
```
