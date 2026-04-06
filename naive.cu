#include <cstdlib>
#include <math.h>
#include <stdio.h>

#define SOFTENING 0.001f
#define DT 0.01f

typedef struct {
  float x, y, z;    // position
  float vx, vy, vz; // velocity
  float mass;
} Body;

__global__ void nbody_naive(Body *bodies, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  float fx = 0.0f, fy = 0.0f, fz = 0.0f;

  for (int j = 0; j < N; j++) {
    if (i == j)
      continue;

    float dx = bodies[j].x - bodies[i].x;
    float dy = bodies[j].y - bodies[i].y;
    float dz = bodies[j].z - bodies[i].z;

    float distSq = dx * dx + dy * dy + dz * dz + SOFTENING * SOFTENING;
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

int main() {
  int N = 16384;
  int nSteps = 10;

  size_t size = N * sizeof(Body);

  // alloc and initialize bodies:
  Body *h_bodies = (Body *)malloc(size);
  for (int i = 0; i < N; i++) {
    h_bodies[i].x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    h_bodies[i].y = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    h_bodies[i].z = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    h_bodies[i].vx = 0.0f;
    h_bodies[i].vy = 0.0f;
    h_bodies[i].vz = 0.0f;
    h_bodies[i].mass = 1.0f;
  }

  // alloc device memory:
  Body *d_bodies;
  cudaMalloc(&d_bodies, size);
  cudaMemcpy(d_bodies, h_bodies, size, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // warmup:
  nbody_naive<<<gridSize, blockSize>>>(d_bodies, N);
  cudaDeviceSynchronize();

  // timing:
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int step = 0; step < nSteps; step++) {
    nbody_naive<<<gridSize, blockSize>>>(d_bodies, N);
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  float msPerStep = ms / nSteps;
  float flops = 20.0f * N * N;
  float gflops = (flops * 1e-9f) / (msPerStep * 1e-3f);

  printf("N = %d\n", N);
  printf("Time per step: %.3f ms\n", msPerStep);
  printf("Performance: %.2f GFLOPS/s\n", gflops);

  cudaMemcpy(h_bodies, d_bodies, size, cudaMemcpyDeviceToHost);
  printf("Body 0 position: (%.3f, %.3f, %.3f)\n", h_bodies[0].x, h_bodies[0].y,
         h_bodies[0].z);

  free(h_bodies);
  cudaFree(d_bodies);
  return 0;
}
