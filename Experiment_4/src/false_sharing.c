#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define N 100000000L

int main(int argc, char **argv) {
    int stride = (argc > 1 && argv[1][0] == 'p') ? 8 : 1;
    int nt = omp_get_max_threads();

    size_t bytes = ((size_t)nt * stride * sizeof(long) + 63) / 64 * 64;

    long *cnt = aligned_alloc(64, bytes);
    if (!cnt) return 1;

    memset(cnt, 0, bytes);

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        volatile long *p = &cnt[omp_get_thread_num() * stride];
        for (long i = 0; i < N / nt; i++)
            (*p)++;
    }

    double elapsed = omp_get_wtime() - start;

    long total = 0;
    for (int i = 0; i < nt; i++)
        total += cnt[i * stride];

    printf("%s | Threads=%d | Total=%ld | Time=%.4f seconds\n",
           stride == 1 ? "Adjacent" : "Padded", nt, total, elapsed);

    free(cnt);
    return 0;
}
