# Binaries Directory

This directory stores compiled executables generated during compilation.

## Build Commands

### On Windows (x64 Native Tools Command Prompt):
```cmd
nvcc -O2 -std=c++14 -arch=sm_89 ..\src\warp_divergence.cu -o warp_divergence.exe
```

### On Linux / Ubuntu:
```bash
nvcc -O2 -std=c++14 -arch=sm_89 ../src/warp_divergence.cu -o warp_divergence
```

*Note: Binaries are typically excluded from git tracking or stored locally during benchmark execution.*
