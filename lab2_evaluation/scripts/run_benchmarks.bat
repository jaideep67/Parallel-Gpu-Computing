@echo off
REM ============================================================================
REM PGC Lab 2: CUDA Warp Divergence Compilation and Benchmark Runner (Windows)
REM ============================================================================

echo ===================================================
echo Step 1: Checking NVIDIA GPU & CUDA Toolchain
echo ===================================================
nvidia-smi
if %ERRORLEVEL% neq 0 (
    echo [ERROR] nvidia-smi failed. Ensure NVIDIA driver is installed.
    exit /b %ERRORLEVEL%
)

nvcc --version
if %ERRORLEVEL% neq 0 (
    echo [ERROR] nvcc not found. Please run from x64 Native Tools Command Prompt for VS.
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo Step 2: Compiling CUDA Warp Divergence Kernel
echo ===================================================
REM Using sm_89 for RTX 4500 Ada Lovelace architecture
nvcc -O2 -std=c++14 -arch=sm_89 src\warp_divergence.cu -o bin\warp_divergence.exe
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compilation failed.
    exit /b %ERRORLEVEL%
)
echo [SUCCESS] Binary created at bin\warp_divergence.exe

echo.
echo ===================================================
echo Step 3: Executing Warp Divergence Benchmarks
echo ===================================================
bin\warp_divergence.exe > results\terminal_output.txt
type results\terminal_output.txt

echo.
echo ===================================================
echo Step 4: Generating Evaluation Graphs
echo ===================================================
python scripts\plot_results.py

echo Benchmark and visualization complete!
