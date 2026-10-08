# data/

No external dataset is needed for this experiment.

Both programs create their input vectors inside the program, in a fixed and repeatable way:

```c
A[i] = 1.0f;   B[i] = 2.0f;   // for every i in 0 .. N-1
```

The vector length `N` is passed on the command line (default 10,000,000), for example `./bin/vector_cuda 25000000`.
The benchmark sizes used in the report are set in `scripts/run_benchmarks.sh`:
`100000, 1000000, 5000000, 10000000, 25000000`.

Because the inputs are fixed, the correct outputs are known exactly: `C_add[i] = 3.00` and `C_mul[i] = 2.00`.

This folder is kept because the course's suggested repository structure includes `data/`.
