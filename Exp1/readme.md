# Experiment 1 – Matrix Multiplication using Sequential, OpenMP, MPI and CUDA

## 📌 Overview

This experiment implements **4000 × 4000 matrix multiplication** using four different computing models:

1. **Sequential Execution**
2. **OpenMP Parallel Execution**
3. **MPI Distributed Execution**
4. **CUDA GPU Execution**

The main objective is to implement the same mathematical problem using different parallel computing approaches and compare their execution performance.

The same matrices are used for all implementations so that the results can be compared fairly.

---

## 🎯 Objectives

- Implement matrix multiplication using sequential CPU execution.
- Parallelize matrix multiplication using **OpenMP**.
- Distribute matrix multiplication across multiple machines using **MPI**.
- Accelerate matrix multiplication using **CUDA** on an NVIDIA GPU.
- Compare execution times of the four implementations.
- Calculate speedup with respect to the sequential implementation.
- Verify that all implementations produce the same output.

---

## 🧮 Problem Definition

The experiment performs multiplication of two matrices:

- Matrix **A**: `4000 × 4000`
- Matrix **B**: `4000 × 4000`
- Result Matrix **C**: `4000 × 4000`

Matrix multiplication is performed using:

\[
C[i][j] = \sum_{k=0}^{3999} A[i][k] \times B[k][j]
\]

For this experiment, all elements of matrices **A** and **B** are initialized to `1.0`.

Therefore:

\[
C[i][j] = 4000.00
\]

The output is verified using:

```text
C[0][0] = 4000.00
```

---

# 🏗️ Computing Models

## 1. Sequential Matrix Multiplication

The sequential implementation performs matrix multiplication using a single CPU execution flow.

It is used as the **baseline** for comparing the performance of OpenMP, MPI and CUDA.

### Characteristics

- Single-threaded CPU execution
- No parallelism
- Simple implementation
- Used as the baseline execution time

### Platform

```text
WSL2 Ubuntu
GCC 15.2.0
Optimization: -O2
```

---

## 2. OpenMP Matrix Multiplication

OpenMP is used to parallelize the matrix multiplication on a shared-memory CPU.

The outer loop iterations are divided among multiple CPU threads.

### Characteristics

- Shared-memory parallelism
- Multiple CPU threads
- Work is divided among threads
- Uses OpenMP compiler support

### Configuration

```text
Platform: WSL2 Ubuntu
Compiler: GCC 15.2.0
Compiler Flag: -O2 -fopenmp
Threads: 8
```

---

## 3. MPI Distributed Matrix Multiplication

MPI is used to distribute the matrix multiplication across multiple processes running on separate Ubuntu virtual machines.

Each process works on a portion of the matrix computation.

### Characteristics

- Distributed-memory parallelism
- Multiple MPI processes
- Communication between processes
- Multiple virtual machines are used as MPI nodes

### Configuration


Nodes: 4 Ubuntu 24.04 VMs
MPI: Open MPI
GCC: 13.3.0
Processes: 4


MPI introduces communication overhead because data must be exchanged between separate processes and virtual machines.

---

## 4. CUDA Matrix Multiplication

CUDA is used to execute matrix multiplication on an NVIDIA GPU.

The CPU prepares the matrices and transfers the data to GPU memory. A CUDA kernel is then launched to perform the matrix multiplication in parallel.

Each logical GPU thread computes an output element of matrix `C`.

### Characteristics

- GPU-based parallel execution
- SIMT (Single Instruction, Multiple Threads) execution
- Large number of parallel threads
- CUDA kernel execution
- Host-to-device and device-to-host memory transfers

### CUDA Configuration


GPU: NVIDIA GPU
Compiler: nvcc
Optimization: -O2
Logical Threads: 16,000,000


For the 4000 × 4000 matrix:


Block size = 16 × 16 = 256 threads
Grid size  = 250 × 250 blocks


Therefore:

```text
250 × 250 × 256 = 16,000,000
```

logical thread instances are launched.

---
## 🔄 Execution Flow

The experiment performs the same `4000 × 4000` matrix multiplication using four different computing models.

```text
                         4000 × 4000 Matrix Multiplication
                                      │
                                      ▼
                         Initialize Matrices A and B
                              (All values = 1.0)
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │      Four Implementations       │
                    └─────────────────────────────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
             ▼                        ▼                        ▼
      Sequential                 OpenMP                    MPI
      Single CPU              8 CPU Threads            4 MPI Processes
             │                        │                        │
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      │
                                      ▼
                                   CUDA
                               NVIDIA GPU
                          Parallel GPU Threads
                                      │
                                      ▼
                         Compute Result Matrix C
                                      │
                                      ▼
                           Verify C[0][0]
                                      │
                                      ▼
                              Expected Result
                                `4000.00`
                                      │
                                      ▼
                         Measure Execution Time
                                      │
                                      ▼
                           Performance Comparison
                                      │
                                      ▼
                    Speedup and Execution-Time Analysis
```
---

# 🛠️ Software and Hardware Requirements

## Hardware

- Windows 10/11 host system
- Sufficient CPU cores and RAM
- Four Ubuntu virtual machines for MPI
- NVIDIA CUDA-capable GPU for CUDA execution

## Software

- Windows PowerShell
- WSL2
- Ubuntu
- GCC / build-essential
- OpenMP
- Open MPI
- OpenSSH
- CUDA Toolkit
- `nvcc`
- VMware Workstation or equivalent virtualization software

---

# 📁 Project Structure
```
Experiment_1/
│
├── README.md
│
├── sequential/
│   └── matrix_multiplication.c
│
├── openmp/
│   └── matrix_multiplication_omp.c
│
├── mpi/
│   ├── matrix_multiplication_mpi.c
│   └── hostfile
│
├── cuda/
│   └── matrix_multiplication.cu
│
├── results/
│   └── performance_results
│
└── screenshots/
    ├── sequential/
    ├── openmp/
    ├── mpi/
    └── cuda/
```


---

# 📊 Performance Results

The same `4000 × 4000` matrix multiplication problem was executed using all four implementations.

All implementations produced the same verification result:


C[0][0] = 4000.00


| Implementation | Model | Platform | Workers | Time (s) | Kernel Time (s) | Verification |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Sequential | Single-thread CPU | WSL2 Ubuntu (GCC 15.2.0 -O2) | 1 | 266.477234 | — | 4000.00 |
| OpenMP | Shared memory | WSL2 Ubuntu (GCC 15.2.0 -O2 -fopenmp) | 8 | 41.021555 | — | 4000.00 |
| MPI | Distributed memory | 4 × Ubuntu 24.04 VMs (Open MPI / GCC 13.3.0) | 4 | 226.167575 | — | 4000.00 |
| CUDA | GPU SIMT | NVIDIA GPU (nvcc -O2 / MSVC x64) | 16000000 | 0.077087 | 0.064499 | 4000.00 |

---

# ⚡ Speedup Analysis

Speedup is calculated using the sequential execution time as the baseline.

### Formula

\[
Speedup = \frac{Sequential\ Time}{Parallel\ Time}
\]

Using the recorded execution times:

| Implementation | Execution Time (s) | Speedup |
| :--- | ---: | ---: |
| Sequential | 266.477234 | 1.00× |
| OpenMP | 41.021555 | 6.50× |
| MPI | 226.167575 | 1.18× |
| CUDA | 0.077087 | 3458.92× |

---

# 📈 Performance Analysis

## Sequential

The sequential implementation takes:

```text
266.477234 seconds
```

It acts as the baseline for calculating speedup.

---

## OpenMP

The OpenMP implementation uses 8 CPU threads and takes:


41.021555 seconds


Compared with the sequential implementation, OpenMP provides approximately:


6.50× speedup


The improvement comes from dividing the matrix computation among multiple CPU threads.

---

## MPI

The MPI implementation uses 4 processes across four Ubuntu virtual machines.

Recorded execution time:

```text
226.167575 seconds
```

The measured speedup compared with the sequential baseline is approximately:

```text
1.18×
```

MPI also involves communication and coordination between separate processes and virtual machines.

---

## CUDA

CUDA executes the matrix multiplication on an NVIDIA GPU.

Recorded values:


Total CUDA Time  = 0.077087 seconds
Kernel Time      = 0.064499 seconds


The measured speedup over the sequential baseline using total CUDA time is approximately:


3458.92×


The CUDA total time includes the CUDA phase involving data transfers and kernel execution.

---

# 📊 Graphs

The following graphs can be used for performance analysis:

### 1. Execution Time Comparison

Compares the execution time of:

- Sequential
- OpenMP
- MPI
- CUDA

Suggested file:


<img width="762" height="461" alt="execution time graph" src="https://github.com/user-attachments/assets/e1479399-2aa8-484b-ae4d-f0de0895fcfa" />



### 2. Speedup Comparison

Shows the speedup of OpenMP, MPI and CUDA relative to sequential execution.

Suggested file:


<img width="756" height="450" alt="Speed Comparsion" src="https://github.com/user-attachments/assets/485b2ed2-df84-4587-b462-bfd3536a470a" />


### 3. CUDA Total Time vs Kernel Time

Compares:

- Total CUDA execution time
- CUDA kernel execution time

Suggested file:


<img width="758" height="501" alt="Total vs kernel time" src="https://github.com/user-attachments/assets/03c23f4b-a08b-4289-a836-ece46b517e65" />


### 4. Performance Improvement

Shows the percentage reduction in execution time relative to the sequential baseline.

Suggested file:


<img width="761" height="453" alt="Performance Graph" src="https://github.com/user-attachments/assets/eb684f3f-d36b-43de-bc17-7442caa4d93a" />



---

# 🔍 Result Verification

All four implementations perform the same mathematical operation and produce:


C[0][0] = 4000.00


This confirms that the matrix multiplication result is consistent across the different computing models.

---

# 📸 Recommended Screenshots

## Sequential

<img width="1920" height="1080" alt="Sequential_mul_wsl" src="https://github.com/user-attachments/assets/c1a1e872-f8d3-4a05-9f26-d8570f9b0f6f" />

Include screenshots showing:

- WSL2 verification
- GCC version
- Source code
- Compilation
- Program execution
- Final output

## OpenMP

<img width="1920" height="1080" alt="OpenMP_matrix_result" src="https://github.com/user-attachments/assets/4bebb475-0106-48bc-81ee-19de2b2273c3" />


Include screenshots showing:

- Number of CPU cores
- `OMP_NUM_THREADS`
- Source code
- CPU/thread activity
- Final output

## MPI

<img width="1484" height="925" alt="mpi_matrixmulti_ping" src="https://github.com/user-attachments/assets/d45be8a0-0699-4dad-9a68-865d9030ad09" />

Include screenshots showing:

- Four Ubuntu virtual machines
- Network connectivity
- SSH verification
- MPI configuration
- `mpirun` execution
- Final output

## CUDA

<img width="1600" height="900" alt="cuda_result" src="https://github.com/user-attachments/assets/86f82d4c-ce32-499b-b687-13529f2c71e6" />

Include screenshots showing:

- NVIDIA GPU detection
- `nvidia-smi`
- CUDA compiler version
- CUDA source code
- Compilation using `nvcc`
- CUDA execution
- Kernel time
- Total CUDA time
- Final verification

---

# 🧠 Comparison of the Four Models

| Feature | Sequential | OpenMP | MPI | CUDA |
| :--- | :--- | :--- | :--- | :--- |
| Execution Type | Single-thread CPU | Multi-thread CPU | Distributed processes | GPU parallel |
| Memory Model | Single memory | Shared memory | Distributed memory | GPU memory + host memory |
| Parallelism | No | CPU threads | MPI processes | GPU threads |
| Workers | 1 | 8 | 4 | 16,000,000 logical threads |
| Main Technology | C/GCC | OpenMP | Open MPI | CUDA |
| Platform | WSL2 Ubuntu | WSL2 Ubuntu | 4 Ubuntu VMs | NVIDIA GPU |

---

# 💡 Key Observations

- Sequential execution provides the baseline performance.
- OpenMP improves performance by distributing loop iterations among multiple CPU threads.
- MPI distributes computation across multiple processes and virtual machines.
- MPI communication introduces additional overhead because processes operate in separate address spaces.
- CUDA uses massive GPU parallelism for matrix multiplication.
- All four implementations produce the same verification value of `4000.00`.
- The recorded execution times demonstrate the performance differences between sequential, shared-memory, distributed-memory and GPU-based execution.

---

# 🎯 Conclusion

This experiment implements the same `4000 × 4000` matrix multiplication problem using four different computing models: **Sequential, OpenMP, MPI and CUDA**.

The sequential implementation provides the baseline execution time. OpenMP parallelizes the computation using multiple CPU threads, while MPI distributes the computation across multiple virtual machines. CUDA executes the computation using massively parallel GPU threads.

The experiment demonstrates how different parallel computing models affect execution time, parallelism, communication overhead and overall performance while maintaining the same mathematical computation and verification result.

---

## 📚 Technologies Used

```text
C
C++
OpenMP
MPI
CUDA
GCC
Open MPI
WSL2
Ubuntu
NVIDIA GPU
```

---

## 👨‍💻 Experiment

**Course:** Parallel Computing Lab (PGC)

**Experiment:** Matrix Multiplication using Sequential, OpenMP, MPI and CUDA

**Matrix Size:** 4000 × 4000

**Parallel Models:**

- Sequential CPU
- OpenMP Shared Memory
- MPI Distributed Memory
- CUDA GPU SIMT

**Verification:** `C[0][0] = 4000.00`
