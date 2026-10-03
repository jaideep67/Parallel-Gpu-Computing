import matplotlib.pyplot as plt

# Number of threads
threads = [1, 2, 4, 6, 16]

# Pthreads execution times
pthreads_time = [
    1.361907,
    0.687491,
    0.357255,
    0.249517,
    0.129455
]

# OpenMP execution times
openmp_time = [
    1.363452,
    0.691519,
    0.360191,
    0.239480,
    0.135777
]

# Sequential execution times
sequential_times = [
    1.368140,
    1.361962
]

# Calculate average sequential time
sequential_time = sum(sequential_times) / len(sequential_times)

print("Average Sequential Time:",
      round(sequential_time, 6), "seconds")


# ==================================================
# 1. EXECUTION TIME GRAPH
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    threads,
    pthreads_time,
    marker='o',
    label='Pthreads'
)

plt.plot(
    threads,
    openmp_time,
    marker='s',
    label='OpenMP'
)

plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time vs Number of Threads")

plt.xticks(threads)
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "../graphs/execution_time_vs_threads.png",
    dpi=300
)

plt.show()


# ==================================================
# 2. SPEEDUP GRAPH
# ==================================================

pthreads_speedup = []

openmp_speedup = []

for i in range(len(threads)):

    pthreads_speedup.append(
        sequential_time / pthreads_time[i]
    )

    openmp_speedup.append(
        sequential_time / openmp_time[i]
    )


plt.figure(figsize=(8, 5))

plt.plot(
    threads,
    pthreads_speedup,
    marker='o',
    label='Pthreads'
)

plt.plot(
    threads,
    openmp_speedup,
    marker='s',
    label='OpenMP'
)

plt.xlabel("Number of Threads")
plt.ylabel("Speedup (×)")
plt.title("Speedup vs Number of Threads")

plt.xticks(threads)
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "../graphs/speedup_vs_threads.png",
    dpi=300
)

plt.show()


# ==================================================
# 3. EFFICIENCY GRAPH
# ==================================================

pthreads_efficiency = []

openmp_efficiency = []

for i in range(len(threads)):

    pthreads_efficiency.append(
        (pthreads_speedup[i] / threads[i]) * 100
    )

    openmp_efficiency.append(
        (openmp_speedup[i] / threads[i]) * 100
    )


plt.figure(figsize=(8, 5))

plt.plot(
    threads,
    pthreads_efficiency,
    marker='o',
    label='Pthreads'
)

plt.plot(
    threads,
    openmp_efficiency,
    marker='s',
    label='OpenMP'
)

plt.xlabel("Number of Threads")
plt.ylabel("Efficiency (%)")
plt.title("Efficiency vs Number of Threads")

plt.xticks(threads)
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "../graphs/efficiency_vs_threads.png",
    dpi=300
)

plt.show()


# ==================================================
# PRINT RESULTS
# ==================================================

print("\nPthreads Speedup:")
for t, s in zip(threads, pthreads_speedup):
    print(t, "threads =", round(s, 2), "x")

print("\nOpenMP Speedup:")
for t, s in zip(threads, openmp_speedup):
    print(t, "threads =", round(s, 2), "x")

print("\nPthreads Efficiency:")
for t, e in zip(threads, pthreads_efficiency):
    print(t, "threads =", round(e, 2), "%")

print("\nOpenMP Efficiency:")
for t, e in zip(threads, openmp_efficiency):
    print(t, "threads =", round(e, 2), "%")

print("\nAll graphs generated successfully!")
