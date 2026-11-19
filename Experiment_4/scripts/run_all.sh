#!/usr/bin/env bash
# Experiment 4 - Cache coherence (MSI / MESI / MOESI), false sharing and NUMA memory access
# Runs every part in the same order as the lab (screenshots 01-06).
# Run inside Ubuntu from the Experiment_4 folder:   bash scripts/run_all.sh
set -e
mkdir -p bin

# ---- 0. system and tools ----------------------------------------------------
# sudo apt update && sudo apt install build-essential numactl hwloc python3 -y
lsb_release -a; nproc; free -h; df -h /
gcc --version | head -1; python3 --version; numactl --version
echo '#include <omp.h>' | gcc -E -fopenmp -xc - > /dev/null && echo "OpenMP OK"
numactl --hardware                         # how many NUMA nodes, which CPUs belong to each
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|Hypervisor"

# ---- Part A: MSI / MESI / MOESI simulation -----------------------------------
python3 src/coherence_sim.py 1000

# ---- Part B1: false sharing, 2 threads (repeat 3 times, keep the median) -----
gcc -O2 src/false_sharing.c -fopenmp -o bin/false_sharing
for run in 1 2 3; do
  OMP_NUM_THREADS=2 ./bin/false_sharing adjacent
  OMP_NUM_THREADS=2 ./bin/false_sharing padded
done

# ---- Part B2: memory sweep, default and node-0 (local) placement -------------
gcc -O2 src/numa_sweep.c -fopenmp -o bin/numa_sweep
for run in 1 2 3; do
  OMP_NUM_THREADS=1 ./bin/numa_sweep 256
  OMP_NUM_THREADS=2 ./bin/numa_sweep 256
done
OMP_NUM_THREADS=2 numactl --cpunodebind=0 --membind=0 ./bin/numa_sweep 256
OMP_NUM_THREADS=1 numactl --cpunodebind=0 --membind=0 ./bin/numa_sweep 256
# remote memory - only possible when numactl --hardware shows 2 or more nodes:
# OMP_NUM_THREADS=2 numactl --cpunodebind=0 --membind=1 ./bin/numa_sweep 256

# ---- comparison program: both tables in one run ------------------------------
gcc -O2 src/compare_numa.c -fopenmp -o bin/compare_numa
./bin/compare_numa | tee results/raw/compare_numa_new_run.txt
