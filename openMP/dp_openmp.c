#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernel_openmp.h"

int main(int argc, char const *argv[]){
	/*Set up CUDA device*/
	/*
	int CUDA_DEVICE = atoi(argv[5]);
	cudaError err = cudaSetDevice(CUDA_DEVICE);
	if(err != cudaSuccess){
		printf("Error setting cuda device\n");
		exit(EXIT_FAILURE);
	}
	*/

	/*Read in data*/
	int rows = atoi(argv[1]);
	int cols = atoi(argv[2]);
	int nprocs = atoi(argv[5]);
	/*allocate memory in host RAM*/

	float* matrix = (float*) malloc(rows * cols * sizeof(float));
	float* vector = (float*) malloc(cols * sizeof(float));
	float* result = (float*) malloc(rows * sizeof(float));
	/*
	cudaMallocHost((void **) &matrix, sizeof(float) * rows * cols);
	cudaMallocHost((void **) &vector, sizeof(float) * cols);
	cudaMallocHost((void **) &result, sizeof(float) * rows);
	*/
	/*Read file*/
	FILE *mp;
	mp = fopen(argv[3], "r");
    int i = 0;
    float matrix_data;
    for(i = 0; i < (rows * cols); i++){
        fscanf(mp, "%f", &matrix_data);
        matrix[i] = matrix_data;
    }

	fclose(mp);
	printf("Read Data-Matrix\n");
	fflush(stdout);

	FILE *vp;
	vp = fopen(argv[4], "r");
	int j = 0;
    float vector_data;
    for(j = 0; j < cols; j++){
        fscanf(vp, "%f", &vector_data);
        vector[j] = vector_data;
    }
	fclose(vp);
	printf("Read Data-Vector\n");
	fflush(stdout);

    /*Allocate memory space on the GPU*/ 
	/*
    float *d_matrix, *d_vector, *d_result;
	err = cudaMalloc((void **) &d_matrix, sizeof(float) * rows * cols);
	if(err != cudaSuccess){
		printf("Error mallocing matrix on GPU device\n");
	}
    err = cudaMalloc((void **) &d_vector, sizeof(float) * cols);
	if(err != cudaSuccess){
		printf("Error mallocing vector on GPU device\n");
	}
	err = cudaMalloc((void **) &d_result, sizeof(float) * rows);
	if(err != cudaSuccess){
		printf("Error mallocing result on GPU device\n");
	}
	*/
    /*copy matrix and vector from host to device memory*/
	/*
    err = cudaMemcpy(d_matrix, matrix, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		printf("Error copying matrix to GPU\n");
	}
	err = cudaMemcpy(d_vector, vector, sizeof(float) * cols, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		printf("Error copying vector to GPU\n");
	}
	*/
	/*int THREADS = atoi(argv[6]);*/
	int jobs;
	/*int BLOCKS;*/
	/*jobs = rows;*/
	jobs = ((rows + nprocs - 1) / nprocs);
	/*BLOCKS = (jobs + THREADS - 1)/THREADS;*/
	/*printf("jobs = %d\n", jobs);*/
#pragma omp parallel num_threads(nprocs)
	/*kernel<<<BLOCKS, THREADS>>>(d_matrix, d_vector, d_result, rows, cols);*/
	kernel(matrix, vector, result, rows, cols, jobs);
	/*copy results from GPU to host*/
	/*
	err = cudaMemcpy(result, d_result, sizeof(float) * rows, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		printf("Error copying results from GPU\n");
	}
    cudaThreadSynchronize();
	*/
	
	/*Print out the result*/
	int k;
	printf("Results :\n");
    for(k = 0; k < rows; k++) {
		printf("%f\n", result[k]);
	}
	printf("\n");
    // free memory
	/*
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
    cudaFreeHost(matrix);
    cudaFreeHost(vector);
    cudaFreeHost(result);
    */
	return 0;
}