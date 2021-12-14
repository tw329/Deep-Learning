#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "kernel.h"

__global__
void kernel(unsigned int N, unsigned int M, unsigned char *mat, unsigned char *vec, float *result){
    // threadIdx.x: The thread id with respect to the thread's block
    //              From 0 - (thread count per block - 1)
     
    // blockIdx.x:  The block id with respect to the grid (all blocks in the kernel)
    //              From 0 - (number of blocks launched - 1)
     
    // blockDim.x:  The number of threads in a block (block's dimension)
    //              Single value
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum;
    if(tid < N){
        printf("Read in data :\nRows : %d\nCols : %d\nMatrix : %s\nVector : %s\n", N, M, mat, vec);
        //printf("tid = %d\n", tid);
        for(int i = 0; i < M; i++){
            //printf("%c * %c\n", vec[i], mat[tid * M + i]);
            //printf("%f * %f\n", (float)(vec[i]) - 48, (float)(mat[tid * M + i]) - 48);
            sum += ((float)(vec[i]) - 48) * ((float)(mat[tid * M + i]) - 48);
        }
        //printf("%f\n", sum);
        result[tid] = sum;
    }
}