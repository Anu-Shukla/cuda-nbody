# CUDA N-Body Simulation

Gravitational N-body simulation in CUDA, optimized across multiple implementation levels and benchmarked on an RTX 4090.

## Results

| Version | N     | Time/step (ms) | GFLOPs/s |
|---------|-------|----------------|----------|
| Naive   | 4096  | 0.251          | ~1,339   |
| Naive   | 8192  | 0.508          | ~2,643   |
| Naive   | 16384 | 1.027          | ~5,227   |
| Tiled   | 4096  | 0.193          | ~1,739   |
| Tiled   | 8192  | 0.383          | ~3,500   |
| Tiled   | 16384 | 0.765          | ~7,017   |

All results use `-O3 --use_fast_math`. Peak RTX 4090 is ~82 TFLOPs.

## Implementations

### 1. Naive Kernel (`naive.cu`)
One thread per particle. Each thread loops over all N particles and accumulates force contributions from global memory. Particle i position cached in registers to avoid repeated global memory reads.

### 2. Shared Memory Tiling (`tiled.cu`)
Threads in a block cooperatively load tiles of TILE_SIZE=256 particles into shared memory. Each thread computes partial forces from the tile before sliding to the next. Reduces global memory traffic by TILE_SIZE. Consistent ~1.34x speedup over naive across all N values tested.

## Key Observations
- GFLOPs/s scales with N as the GPU becomes better utilized at larger problem sizes
- `--use_fast_math` gives ~2x speedup by replacing `sqrtf` with a faster approximate version — worthwhile since N-body already uses a softening approximation
- Tiled speedup plateaus at ~1.34x suggesting we remain memory bandwidth bound
