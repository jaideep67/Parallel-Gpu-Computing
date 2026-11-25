#!/usr/bin/env bash
# ==============================================================================
# PGC Lab 2: CUDA Warp Divergence Benchmark Runner (Linux/Unix)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "=== Step 1: Checking GPU & Toolchain ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "Warning: nvidia-smi not found."
fi

nvcc --version

echo ""
echo "=== Step 2: Compiling CUDA Kernel ==="
mkdir -p bin
# Auto-detect or default to native / sm_89
ARCH_FLAG="-arch=native"
if ! nvcc --help | grep -q -- "-arch=native"; then
    ARCH_FLAG="-arch=sm_89"
fi

nvcc -O2 -std=c++14 "$ARCH_FLAG" src/warp_divergence.cu -o bin/warp_divergence
echo "[OK] Built bin/warp_divergence"

echo ""
echo "=== Step 3: Running Benchmark ==="
mkdir -p results
./bin/warp_divergence | tee results/terminal_output.txt

echo ""
echo "=== Step 4: Generating Visualization Plots ==="
python3 scripts/plot_results.py

echo "Benchmarking and graph generation completed successfully."
