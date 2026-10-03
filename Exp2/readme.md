# Experiment 2 — Multithreaded Programming using Pthreads and OpenMP

## 1. Experiment Title

**Develop Multithreaded Programs Using Parallel Programming Libraries to Understand Thread Creation, Management, and Coordination**

---

## 2. Aim

The aim of this experiment is to develop multithreaded programs using **Pthreads** and **OpenMP** and understand:

- Thread creation
- Thread management
- Work distribution
- Race conditions
- Synchronisation
- Thread coordination
- Performance improvement using multiple threads

A thread can be considered as a worker that performs a part of a program's work. Multithreading allows multiple workers to execute different parts of a problem concurrently. 

---

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| C | Programming language |
| Pthreads | Explicit thread creation and management |
| OpenMP | High-level parallel programming |
| GCC | C compiler |
| WSL Ubuntu | Linux execution environment |
| Nano | Text editor |

---

## 4. Experiment Overview

This experiment is divided into three major parts:

### Part A — Pthreads

Pthreads (POSIX Threads) provides explicit control over thread creation, execution, joining, and synchronisation.

Important functions used:

- `pthread_create()`
- `pthread_join()`
- `pthread_mutex_lock()`
- `pthread_mutex_unlock()`

- <img width="1920" height="1080" alt="pthread_step_1_5" src="https://github.com/user-attachments/assets/f12b894d-deb9-432d-ae96-13794355d665" />


### Part B — OpenMP

OpenMP provides a higher-level programming model for parallel programming using compiler directives.

Important concepts used:

- Parallel regions
- Work sharing
- Reduction
- Critical sections
- Barriers
- Thread identification

- <img width="1920" height="1080" alt="Openmp_step_6_7" src="https://github.com/user-attachments/assets/2a99b0c2-2a0e-4db0-896b-9c142db99c2d" />


### Part C — Performance Analysis

The performance of sequential, Pthreads, and OpenMP implementations is analysed using:

<img width="1920" height="1080" alt="OpenMP vs Pthreads " src="https://github.com/user-attachments/assets/966a9619-d2aa-456c-ba53-0d87a2e3cbe6" />


- Execution time
- Speedup
- Efficiency
- Graphical comparison

---

<img width="759" height="472" alt="Execution time" src="https://github.com/user-attachments/assets/4ed0a9ee-87eb-4149-af8a-01a2991928c0" />

<img width="756" height="474" alt="Speed Vs No of threads" src="https://github.com/user-attachments/assets/341e896b-bf2e-4bc9-8957-7ed6ae407295" />

<img width="756" height="474" alt="Efficiency Graph" src="https://github.com/user-attachments/assets/6199b674-60f9-43af-9886-04780e2a12a1" />


## 5. Programs Implemented

### Pthreads Programs

| Program | Description |
|---|---|
| `thread1.c` | Creates and executes one thread |
| `thread2.c` | Creates multiple threads |
| `thread_sum.c` | Divides work among multiple threads |
| `race.c` | Demonstrates a race condition |
| `mutex.c` | Fixes the race condition using a mutex |
| `pthread_perf.c` | Measures Pthreads execution performance |

### OpenMP Programs

| Program | Description |
|---|---|
| `omp1.c` | Demonstrates parallel regions and thread identification |
| `omp_sum.c` | Demonstrates work sharing and reduction |
| `omp_race.c` | Demonstrates a race condition |
| `omp_critical.c` | Fixes the race condition using `critical` |
| `omp_barrier.c` | Demonstrates thread coordination using a barrier |
| `omp_perf.c` | Measures OpenMP execution performance |

---

## 6. Race Condition and Synchronisation

A **race condition** occurs when multiple threads access or modify shared data at the same time without proper coordination.

The result may become incorrect because the threads interfere with each other's updates.



### Pthreads

A mutex is used to protect the critical section.

```text
race.c
   ↓
No Lock
   ↓
Incorrect Result

mutex.c
   ↓
pthread_mutex
   ↓
Correct Result
