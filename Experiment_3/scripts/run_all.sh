#!/usr/bin/env bash
# Experiment 3 - Synchronization mechanisms in parallel programs
# Compiles and runs every program, in the same order as the lab (screenshots 01-04).
# Run inside Ubuntu from the Experiment_3 folder:   bash scripts/run_all.sh
set -e
mkdir -p bin

# check that GCC accepts OpenMP
echo '#include <omp.h>' | gcc -E -fopenmp -x c - > /dev/null && echo "OpenMP OK"
nproc                                              # logical CPUs = default OpenMP thread count

gcc src/race.c         -fopenmp -o bin/race         && ./bin/race          # wrong count (race)
gcc src/critical.c     -fopenmp -o bin/critical     && ./bin/critical      # 10000000
gcc src/atomic.c       -fopenmp -o bin/atomic       && ./bin/atomic        # 10000000
gcc src/barrier.c      -fopenmp -o bin/barrier      && ./bin/barrier       # all "before" lines first
gcc src/semaphore.c    -pthread -o bin/semaphore    && ./bin/semaphore     # 200000

# benchmark: None / Critical / Atomic / Lock with 1, 2, 4 and 8 threads
gcc src/compare_sync.c -fopenmp -o bin/compare_sync && ./bin/compare_sync | tee results/raw/compare_sync_new_run.txt

# optional: run any program with a chosen thread count
# OMP_NUM_THREADS=8 ./bin/critical
