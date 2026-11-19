/* Program 4 - Barrier
 * No thread prints "after barrier" until every thread has printed "before barrier".
 * Compile: gcc barrier.c -fopenmp -o barrier
 */
#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        printf("Thread %d before barrier\n", omp_get_thread_num());
        #pragma omp barrier
        printf("Thread %d after barrier\n", omp_get_thread_num());
    }
}
