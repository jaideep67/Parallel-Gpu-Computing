/* Program 5 - Semaphore (POSIX)
 * A binary semaphore (initial value 1) lets one thread at a time update the counter.
 * Compile: gcc semaphore.c -pthread -o semaphore
 */
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

sem_t semaphore;
int counter = 0;

void *work(void *a) {
    for (int i = 0; i < 100000; i++) {
        sem_wait(&semaphore);     /* value 1 -> 0 (or wait if already 0) */
        counter++;
        sem_post(&semaphore);     /* value 0 -> 1, wakes a waiting thread */
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;
    sem_init(&semaphore, 0, 1);
    pthread_create(&t1, NULL, work, NULL);
    pthread_create(&t2, NULL, work, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("Counter=%d\n", counter);
    sem_destroy(&semaphore);
}
