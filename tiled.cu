#include <cstdlib>
#include <math.h>
#include <stdio.h>

#define SOFTENING 0.001f
#define DT 0.01f
#define TILE_SIZE 256

typedef struct {
  float x, y, z;
  float vx, vy, vz;
  float mass;
} Body;

__global__ void nbody_tiled(Body *bodies, int N) {
  __shared__ Body tile[TILE_SIZE];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float fx = 0.0f, fy = 0.0f, fz = 0.0f;

  float xi = bodies[i].x;
  float yi = bodies[i].y;
  float zi = bodies[i].z;

  for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
    // cooperative load
    int j = t * TILE_SIZE + threadIdx.x;
    if (j < N) {
      tile[threadIdx.x] = bodies[j];
    }
    __syncthreads();

    for (int k = 0; k < TILE_SIZE && (t * TILE_SIZE + k) < N; k++) {
      if (i == t * TILE_SIZE + k)
        continue;

      float dx = tile[k].x - xi;
      float dy = tile[k].y - yi;
      float dz = tile[k].z - zi;

      float distSq = dx * dx + dy * dy + dz * dz + SOFTENING * SOFTENING;
      float invDist = 1.0f / sqrtf(distSq);
      float invDist3 = invDist * invDist * invDist;

      fx += tile[k].mass * dx * invDist3;
      fy += tile[k].mass * dy * invDist3;
      fz += tile[k].mass * dz * invDist3;
    }
    __syncthreads();
  }

  if (i < N) {
    bodies[i].vx += fx * DT;
    bodies[i].vy += fy * DT;
    bodies[i].vz += fz * DT;
  }
}

int main() {
  int N = 16384;
  int nSteps = 10;
  size_t size = N * sizeof(Body);

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

  Body *d_bodies;
  cudaMalloc(&d_bodies, size);
  cudaMemcpy(d_bodies, h_bodies, size, cudaMemcpyHostToDevice);

  int blockSize = TILE_SIZE;
  int gridSize = (N + blockSize - 1) / blockSize;

  // warmup
  nbody_tiled<<<gridSize, blockSize>>>(d_bodies, N);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int step = 0; step < nSteps; step++) {
    nbody_tiled<<<gridSize, blockSize>>>(d_bodies, N);
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
  printf("Performance: %.2f GFLOPs/s\n", gflops);

  cudaMemcpy(h_bodies, d_bodies, size, cudaMemcpyDeviceToHost);
  printf("Body 0 position: (%.3f, %.3f, %.3f)\n", h_bodies[0].x, h_bodies[0].y,
         h_bodies[0].z);

  free(h_bodies);
  cudaFree(d_bodies);
  return 0;
}
