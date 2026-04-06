#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <omp.h>

#define SOFTENING 0.001f
#define DT 0.01f

typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float mass;
} Body;

void nbody_cpu(Body* bodies, int N) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;

        for (int j = 0; j < N; j++) {
            if (i == j) continue;

            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;

            float distSq = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
            float invDist = 1.0f / sqrtf(distSq);
            float invDist3 = invDist * invDist * invDist;

            fx += bodies[j].mass * dx * invDist3;
            fy += bodies[j].mass * dy * invDist3;
            fz += bodies[j].mass * dz * invDist3;
        }

        bodies[i].vx += fx * DT;
        bodies[i].vy += fy * DT;
        bodies[i].vz += fz * DT;
    }
}

int main() {
    int N = 16384;
    int nSteps = 10;

    Body* bodies = (Body*)malloc(N * sizeof(Body));
    for (int i = 0; i < N; i++) {
        bodies[i].x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        bodies[i].y = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        bodies[i].z = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        bodies[i].vx = 0.0f;
        bodies[i].vy = 0.0f;
        bodies[i].vz = 0.0f;
        bodies[i].mass = 1.0f;
    }

    // warmup
    nbody_cpu(bodies, N);

    double start = omp_get_wtime();
    for (int step = 0; step < nSteps; step++) {
        nbody_cpu(bodies, N);
    }
    double end = omp_get_wtime();

    double msPerStep = (end - start) * 1000.0 / nSteps;
    double flops = 20.0 * N * N;
    double gflops = (flops * 1e-9) / (msPerStep * 1e-3);

    printf("N = %d\n", N);
    printf("Cores: %d\n", omp_get_max_threads());
    printf("Time per step: %.3f ms\n", msPerStep);
    printf("Performance: %.2f GFLOPs/s\n", gflops);
    printf("Body 0 position: (%.3f, %.3f, %.3f)\n",
           bodies[0].x, bodies[0].y, bodies[0].z);

    free(bodies);
    return 0;
}
