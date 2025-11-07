/*
Compile with:
gcc -O3 -mavx2 -mfma -fopenmp -ffast-math main.c -o main -lm
*/
/*

## Speedup from Unoptimized Version
| Optimization                                         | Speedup      |
|------------------------------------------------------|--------------|
| Parallelize outer loop (matmul)                      | 4.46 ± 0.85  |
| Parallelize outer loop (softmax)                     | no speedup   |
| Parallelize outer loop (maxindex)                    | no speedup   |
| Memory alignment of all matrices                     | 8.09 ± 0.17  |
| Memory alignment + parallel matmul                   | 3.01 ± 0.43  |
| Memory alignment + SIMD in matmul                    | 8.20 ± 0.25  |
| Memory alignment + SIMD in matmul + parallel matmul  | 28.45 ± 3.41 |

## Speedup from Version: Memory Alignment + SIMD in matmul + Parallel Matmul
Optimizations are cumulative

| Optimization                                         | Speedup      |
|------------------------------------------------------|--------------|
| Compute matmul in blocks (cache locality)            | 1.45 ± 0.25  |
| Compute matmul in blocks + prefetching               | 1.43 ± 0.22  |

## Total Speedup
The current version has a speeup of 45 ± 7 to the unoptimized version.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

static const int N = 1024;

void matmul_unoptimized(float* restrict A, float* restrict B, float* restrict C) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for(int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void softmax_unoptimized(float* restrict C, float* restrict D) {
    for(int i = 0; i < N; i++) {
        float max = C[i * N];
        for(int j = 1; j < N; j++) {
            if(C[i * N + j] > max) {
                max = C[i * N + j];
            }
        }
        for(int j = 0; j < N; j++) {
            D[i * N + j] = expf(C[i * N + j] - max);
        }
        float sum = 0;
        for(int j = 0; j < N; j++) {
            sum += D[i * N + j];
        }
        for(int j = 0; j < N; j++) {
            D[i * N + j] /= sum;
        }
    }
}

void maxindex_unoptimized(float* restrict D, int* restrict E) {
    for(int i = 0; i < N; i++) {
        int max_index = 0;
        float max_value = D[i * N];
        for(int j = 1; j < N; j++) {
            if(D[i * N + j] > max_value) {
                max_value = D[i * N + j];
                max_index = j;
            }
        }
        E[i] = max_index;
    }
}

void matmul_par_avx(float* restrict A, float* restrict B, float* restrict C) {
    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j += 8) {
            __m256 sum = _mm256_setzero_ps();

            for(int k = 0; k < N; k++) {
                __m256 a = _mm256_set1_ps(A[i * N + k]);
                __m256 b = _mm256_load_ps(&B[k * N + j]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }

            _mm256_store_ps(&C[i * N + j], sum);
        }
    }
}

void matmul(float* restrict A, float* restrict B, float* restrict C) {
    // Clear output matrix
    #pragma omp parallel for
    for(int i = 0; i < N * N; i += 8) {
        _mm256_store_ps(&C[i], _mm256_setzero_ps());
    }

    // Block sizes tuned for L1/L2 cache
    const int BLOCK_I = 64;
    const int BLOCK_J = 256;
    const int BLOCK_K = 32;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int ii = 0; ii < N; ii += BLOCK_I) {
        for(int jj = 0; jj < N; jj += BLOCK_J) {
            // Local buffer for B block to improve cache locality
            float B_block[BLOCK_K * BLOCK_J] __attribute__((aligned(32)));

            for(int kk = 0; kk < N; kk += BLOCK_K) {
                // Copy B block for better locality (B is accessed column-wise)
                int k_max = (kk + BLOCK_K < N) ? kk + BLOCK_K : N;
                int j_max = (jj + BLOCK_J < N) ? jj + BLOCK_J : N;

                // Transpose B block for sequential access
                for(int k = kk; k < k_max; k++) {
                    for(int j = jj; j < j_max; j++) {
                        B_block[(k - kk) * BLOCK_J + (j - jj)] = B[k * N + j];
                    }
                }

                // Process A block with vectorization
                int i_max = (ii + BLOCK_I < N) ? ii + BLOCK_I : N;

                for(int i = ii; i < i_max; i++) {
                    float* C_row = &C[i * N];

                    for(int k = kk; k < k_max; k++) {
                        float a_val = A[i * N + k];
                        __m256 a_vec = _mm256_set1_ps(a_val);

                        float* B_block_row = &B_block[(k - kk) * BLOCK_J];

                        // Process 8 elements at a time with AVX2
                        int j = jj;
                        for(; j <= j_max - 8; j += 8) {
                            __m256 b_vec = _mm256_load_ps(&B_block_row[j - jj]);
                            __m256 c_vec = _mm256_load_ps(&C_row[j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_store_ps(&C_row[j], c_vec);
                        }

                        // Handle remaining elements
                        for(; j < j_max; j++) {
                            C_row[j] += a_val * B_block_row[j - jj];
                        }
                    }
                }
            }
        }
    }
}

void softmax(float* C, float* D) {
    for(int i = 0; i < N; i++) {
        float max = C[i * N];
        for(int j = 1; j < N; j++) {
            if(C[i * N + j] > max) {
                max = C[i * N + j];
            }
        }
        for(int j = 0; j < N; j++) {
            D[i * N + j] = expf(C[i * N + j] - max);
        }
        float sum = 0;
        for(int j = 0; j < N; j++) {
            sum += D[i * N + j];
        }
        for(int j = 0; j < N; j++) {
            D[i * N + j] /= sum;
        }
    }
}

void maxindex(float* restrict D, int* restrict E) {
    for(int i = 0; i < N; i++) {
        int max_index = 0;
        float max_value = D[i * N];
        for(int j = 1; j < N; j++) {
            if(D[i * N + j] > max_value) {
                max_value = D[i * N + j];
                max_index = j;
            }
        }
        E[i] = max_index;
    }
}

void compute(float* A, float* B, int* E) {
    float* C = aligned_alloc(64, N * N * sizeof(float));
    float* D = aligned_alloc(64, N * N * sizeof(float));
    // float* C = malloc(N * N * sizeof(float));
    // float* D = malloc(N * N * sizeof(float));

    matmul(A, B, C);
    softmax(C, D);
    maxindex(D, E);

    free(C);
    free(D);
}

int main() {
    float* A = aligned_alloc(64, N * N * sizeof(float));
    float* B = aligned_alloc(64, N * N * sizeof(float));
    int* E   = aligned_alloc(64, N * sizeof(int));
    // float* A = malloc(N * N * sizeof(float));
    // float* B = malloc(N * N * sizeof(float));
    // int* E   = malloc(N * sizeof(int));

    unsigned int seed = 42;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            seed = seed * 1103515245 + 12345;
            A[i * N + j] = (float)(seed % 1000) / 1000.0f - 0.5f;
            seed = seed * 1103515245 + 12345;
            B[i * N + j] = (float)(seed % 1000) / 1000.0f - 0.5f;
        }
    }

    compute(A, B, E);

    long long sum = 0;
    for(int i = 0; i < N; i++) {
        sum += E[i];
    }
    printf("Sum of max indices (per softmax row): %lld\n", sum);

    free(A);
    free(B);
    free(E);
    return 0;
}

