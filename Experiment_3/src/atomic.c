/* Program 3 - Atomic operation
 * The increment is done by one indivisible hardware instruction instead of a lock.
 * Compile: gcc atomic.c -fopenmp -o atomic
 */
#include <stdio.h>
#include <omp.h>

int main() {
    long counter = 0;
    #pragma omp parallel for
    for (long i = 0; i < 10000000; i++) {
        #pragma omp atomic
        counter++;
    }
    printf("Counter=%ld\n", counter);
}
