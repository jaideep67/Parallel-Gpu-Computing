#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define N 100000000L
#define MB 256UL

double counters(int nt, int stride, long *total) {
    size_t bytes = ((size_t)nt * stride * sizeof(long) + 63) / 64 * 64;
    long *cnt = aligned_alloc(64, bytes);
    if (!cnt) exit(1);

    memset(cnt, 0, bytes);
    double start = omp_get_wtime();

    #pragma omp parallel num_threads(nt)
    {
        volatile long *p = &cnt[omp_get_thread_num() * stride];
        for (long i = 0; i < N / nt; i++)
            (*p)++;
    }

    double elapsed = omp_get_wtime() - start;
    *total = 0;

    for (int i = 0; i < nt; i++)
        *total += cnt[i * stride];

    free(cnt);
    return elapsed;
}

double sweep(char *buf, size_t n, int nt) {
    long sum = 0;
    double start = omp_get_wtime();

    for (int pass = 0; pass < 4; pass++) {
        #pragma omp parallel for num_threads(nt) reduction(+:sum)
        for (size_t i = 0; i < n; i += 64)
            sum += buf[i];
    }

    return omp_get_wtime() - start;
}

int main(void) {
    int threads[] = {1, 2};

    printf("FALSE SHARING RESULTS\n");
    printf("Layout   Threads  Total       Time\n");

    for (int i = 0; i < 2; i++) {
        for (int stride = 1; stride <= 8; stride *= 8) {
            long total;
            double t = counters(threads[i], stride, &total);
            printf("%-8s %7d %11ld %.4f seconds\n",
                   stride == 1 ? "Adjacent" : "Padded",
                   threads[i], total, t);
        }
    }

    size_t n = MB << 20;
    char *buf = malloc(n);
    if (!buf) return 1;
    memset(buf, 1, n);

    printf("\nMEMORY SWEEP RESULTS (%lu MB)\n", MB);
    printf("Threads  Time\n");

    for (int i = 0; i < 2; i++) {
        double t = sweep(buf, n, threads[i]);
        printf("%7d  %.4f seconds\n", threads[i], t);
    }

    free(buf);
    return 0;
}
