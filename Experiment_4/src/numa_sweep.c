#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? atol(argv[1]) : 256;
    size_t n = mb << 20;

    char *buf = malloc(n);
    if (!buf) return 1;

    for (size_t i = 0; i < n; i++)
        buf[i] = 1;

    long sum = 0;
    double start = omp_get_wtime();

    for (int pass = 0; pass < 4; pass++) {
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < n; i += 64)
            sum += buf[i];
    }

    double elapsed = omp_get_wtime() - start;

    printf("Buffer=%zu MB | Threads=%d | Sum=%ld | Time=%.4f seconds\n",
           mb, omp_get_max_threads(), sum, elapsed);

    free(buf);
    return 0;
}
