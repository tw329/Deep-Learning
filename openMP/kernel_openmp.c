#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "kernel_openmp.h"

void kernel(float *matrix, float *vector, float *result, int rows, int cols, int jobs){ 
    /*
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    */
    int stop;
    int i, j;
    int tid = omp_get_thread_num();
    if ((tid+1)*jobs > rows){
        stop = rows;
    }
    else{
        stop = (tid+1) * jobs;
    }
    /*printf("thread_id = %d\nstart = %d\nstop = %d\n", tid, tid * jobs, stop);*/
    float sum = 0;
    for (j = tid * jobs; j < stop; j++){
        if( j < rows){
            //printf("%d\n", tid);
            //printf("Matrix : %f\nVector : %f\nRows : %d\nCols : %d\n", matrix, vector, rows, cols);
            for(i = 0; i < cols; i++){
                //printf("%f * %f\n", matrix[tid * cols + i], vector[i]);
                sum += matrix[j * cols + i] * vector[i];
            }
            result[j] = sum;
            sum = 0;
        }
    }
} 