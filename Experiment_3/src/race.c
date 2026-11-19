/* Program 1 - Race condition (no synchronization)
 * Many threads increment one shared counter with no protection, so updates are lost.
 * Compile: gcc race.c -fopenmp -o race
 */
#include <stdio.h>
#include <omp.h>

int main() {
    long counter = 0;
    #pragma omp parallel for
    for (long i = 0; i < 10000000; i++) counter++;   /* unsafe read -> add -> write */
    printf("Counter=%ld\n", counter);
}
