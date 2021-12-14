#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "kernel.h"

__global__
void kernel(float *matrix, float *vector, float *result, int rows, int cols){ 
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    float sum = 0;
    if( tid < rows){
        //printf("%d\n", tid);
        //printf("Matrix : %f\nVector : %f\nRows : %d\nCols : %d\n", matrix, vector, rows, cols);
        for(int i = 0; i < cols; i++){
            //printf("%f * %f\n", matrix[tid * cols + i], vector[i]);
            sum += matrix[tid * cols + i] * vector[i];
        }
        result[tid] = sum;
    }
} 