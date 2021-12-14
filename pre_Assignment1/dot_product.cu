#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include "kernel.h"

int main(int argc, char* argv[]){
	FILE *mp;
	FILE *vp;
	size_t size_m, size_v;
    // Perform matrix multiplication C = A*B
	// A is a rows*cols matrix 
	// B is a 1*cols vector
	// C is rows*1 vector
    unsigned int rows=atoi(argv[1]);
	unsigned int cols=atoi(argv[2]);
	int CUDA_DEVICE = atoi(argv[5]);
	int THREADS = atoi(argv[6]);
	printf("Rows=%d Cols=%d CUDA_DEVICE=%d\n", rows, cols, CUDA_DEVICE);


	// set CUDA device
	cudaError err = cudaSetDevice(CUDA_DEVICE);
	if(err != cudaSuccess) { printf("Error setting CUDA DEVICE\n"); exit(EXIT_FAILURE); }

	// Variable Declaration
	int BLOCKS;
	float* host_results=(float*) malloc(rows * sizeof(float));
	unsigned int jobs;
	unsigned long i;

	unsigned char *dev_matrixT, *dev_vectorT;
	float *results;
	char *line_m = NULL, *line_v = NULL;
	size_t len_m = 0, len_v = 0;
	char *token_m, *token_v, *saveptr_m, *saveptr_v;

	//Read File-Matrix
	mp = fopen(argv[3], "r");
	if (mp == NULL){
		printf("Fail to open file-matrix");
		return 0;
	}

	size_m = (size_t)((size_t)rows * (size_t)cols);
	unsigned char *matrixT = (unsigned char*)malloc((size_t)size_m);
	i = 0;
	while (getline(&line_m, &len_m, mp) != -1){
		token_m = strtok_r(line_m, " ", &saveptr_m);
		while (token_m != NULL){
			matrixT[i] = *token_m;
			i = i + strlen(matrix[i]);
			token_m = strtok_r(NULL, " ", &saveptr_m);
		}
	}
	fclose(mp);
	printf("Read Data-Matrix\n");
	fflush(stdout);

	//Read File-Vector
	vp = fopen(argv[4], "r");
	if (vp == NULL){
		printf("Fail to open file-Vector");
		return 0;
	}

	size_v = (size_t)((size_t)cols);
	unsigned char *vectorT = (unsigned char*)malloc((size_t)size_v);
	i = 0;
	while (getline(&line_v, &len_v, vp) != -1){
		token_v = strtok_r(line_v, " ", &saveptr_v);
		while (token_v != NULL){
			vectorT[i] = *token_v;
			i = i + 1;
			token_v = strtok_r(NULL, " ", &saveptr_v);
		}
	}
	fclose(vp);
	printf("Read Data-Vector\n");
	fflush(stdout);

	//allocate memory in GPU
	err = cudaMalloc((unsigned char**) &dev_matrixT, (size_t) size_m * (size_t) sizeof(unsigned char));
	if(err != cudaSuccess){
		printf("Error mallocing matrix on GPU device\n");
	}
	
	err = cudaMalloc((unsigned char**) &dev_vectorT, (size_t) size_v * (size_t) sizeof(unsigned char));
	if(err != cudaSuccess){
		printf("Error mallocing vector on GPU device\n");
	}

	err = cudaMalloc((float**) &results, rows * sizeof(float));
	if(err != cudaSuccess){
		printf("Error mallocing results on GPU device\n");
	}
	
	//Copy data to GPU
	err = cudaMemcpy(dev_matrixT, matrixT, (size_t)size_m * (size_t)sizeof(unsigned char), cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		printf("Error copying matrix to GPU\n");
	}

	err = cudaMemcpy(dev_vectorT, vectorT, (size_t)size_v * (size_t)sizeof(unsigned char), cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		printf("Error copying vector to GPU\n");
	}

	jobs = rows;
	BLOCKS = (jobs + THREADS - 1)/THREADS;

	kernel<<<BLOCKS,THREADS>>>(rows, cols, dev_matrixT, dev_vectorT, results);
	//copy results to CPU from GPU
	err = cudaMemcpy(host_results, results, rows * sizeof(float), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		printf("Error copying results from GPU\n");
	}

	for(int k = 0; k < jobs; k++) {
		printf("%f ", host_results[k]);
	}
	printf("\n");

	cudaFree(dev_matrixT);
	cudaFree(dev_vectorT);
	cudaFree(results);

	return 0;
}
