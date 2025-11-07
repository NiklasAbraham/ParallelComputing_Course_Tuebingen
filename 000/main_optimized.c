#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const int N = 1024;

void matmul(float* restrict A, float* restrict B, float* restrict C) {
    // IKJ loop order for better cache locality
    for(int i = 0; i < N; i++) {
        for(int k = 0; k < N; k++) {
            float a_ik = A[i * N + k];
            for(int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

void softmax(float* restrict C, float* restrict D) {
    for(int i = 0; i < N; i++) {
        const int row_offset = i * N;
        float max = C[row_offset];
        
        // Find max and compute exp in single pass
        for(int j = 1; j < N; j++) {
            if(C[row_offset + j] > max) {
                max = C[row_offset + j];
            }
        }
        
        float sum = 0.0f;
        for(int j = 0; j < N; j++) {
            float val = expf(C[row_offset + j] - max);
            D[row_offset + j] = val;
            sum += val;
        }
        
        // Normalize
        float inv_sum = 1.0f / sum;
        for(int j = 0; j < N; j++) {
            D[row_offset + j] *= inv_sum;
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

void compute(float* restrict A, float* restrict B, int* restrict E) {
    float* C = malloc(N * N * sizeof(float));
    float* D = malloc(N * N * sizeof(float));
    
    // Initialize C to zero
    for(int i = 0; i < N * N; i++) {
        C[i] = 0.0f;
    }

    matmul(A, B, C);
    softmax(C, D);
    maxindex(D, E);

    free(C);
    free(D);
}

int main() {
    float* A = malloc(N * N * sizeof(float));
    float* B = malloc(N * N * sizeof(float));
    int* E   = malloc(N * sizeof(int));

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

