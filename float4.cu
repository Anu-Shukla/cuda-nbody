#include <cstdlib>
#include <math.h>
#include <stdio.h>

#define SOFTENING 0.001f
#define DT 0.01f
#define TILE_SIZE 256

__global__ void nbody_float4(float4 *pos, float4 *vel, int N) {
  __shared__ float4 tile[TILE_SIZE];

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float4 pi = pos[i];
  float fx = 0.0f, fy = 0.0f, fz = 0.0f;

  for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
    int j = t * TILE_SIZE + threadIdx.x;
    if (j < N) {
      tile[threadIdx.x] = pos[j];
    }
    __syncthreads();

    for (int k = 0; k < TILE_SIZE && (t * TILE_SIZE + k) < N; k++) {
      if (i == t * TILE_SIZE + k)
        continue;

      float dx = tile[k].x - pi.x;
      float dy = tile[k].y - pi.y;
      float dz = tile[k].z - pi.z;

      float distSq = dx * dx + dy * dy + dz * dz + SOFTENING * SOFTENING;
      float invDist = 1.0f / sqrtf(distSq);
      float invDist3 = invDist * invDist * invDist;

      float mass = tile[k].w;
      fx += mass * dx * invDist3;
      fy += mass * dy * invDist3;
      fz += mass * dz * invDist3;
    }
    __syncthreads();
  }

  if (i < N) {
    vel[i].x += fx * DT;
    vel[i].y += fy * DT;
    vel[i].z += fz * DT;
  }
}

int main() {
  int N = 4096;
  int nSteps = 10;

  size_t posSize = N * sizeof(float4);
  size_t velSize = N * sizeof(float4);

  // allocate host memory
  float4 *h_pos = (float4 *)malloc(posSize);
  float4 *h_vel = (float4 *)malloc(velSize);

  // initialize
  for (int i = 0; i < N; i++) {
    h_pos[i].x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    h_pos[i].y = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    h_pos[i].z = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    h_pos[i].w = 1.0f; // mass
    h_vel[i].x = 0.0f;
    h_vel[i].y = 0.0f;
    h_vel[i].z = 0.0f;
    h_vel[i].w = 0.0f;
  }

  // allocate device memory
  float4 *d_pos, *d_vel;
  cudaMalloc(&d_pos, posSize);
  cudaMalloc(&d_vel, velSize);

  cudaMemcpy(d_pos, h_pos, posSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vel, h_vel, velSize, cudaMemcpyHostToDevice);

  int blockSize = TILE_SIZE;
  int gridSize = (N + blockSize - 1) / blockSize;

  // warmup
  nbody_float4<<<gridSize, blockSize>>>(d_pos, d_vel, N);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int step = 0; step < nSteps; step++) {
    nbody_float4<<<gridSize, blockSize>>>(d_pos, d_vel, N);
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

  cudaMemcpy(h_pos, d_pos, posSize, cudaMemcpyDeviceToHost);
  printf("Body 0 position: (%.3f, %.3f, %.3f)\n", h_pos[0].x, h_pos[0].y,
         h_pos[0].z);

  free(h_pos);
  free(h_vel);
  cudaFree(d_pos);
  cudaFree(d_vel);
  return 0;
}
