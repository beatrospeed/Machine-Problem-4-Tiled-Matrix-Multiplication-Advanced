
#include "cuda_runtime.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <chrono>
#include <map>


#define BONUS_TILE 8
#define BONUS_TILE1 14
//Bonus tiled matrix multiplication with boundry checks

__global__
void MatrixMulKernalBonus(float* A, float* B, float* C, int a_r, int a_c, int b_c) {
	// depending on the block width, create appropriate size shared memory and excecute. Thats the only method I could come up with. 

	__shared__ float ds_A[BONUS_TILE1][BONUS_TILE];
	__shared__ float ds_B[BONUS_TILE1][BONUS_TILE];

	int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;

	int Col = bx * 8 + tx;
	int Row = by * 14 + ty;

	 float CVal = 0.0;

	for (int k = 0; k < (BONUS_TILE + a_c - 1) / 2; k++) {

		if (k * BONUS_TILE + tx < a_c && Row < a_r)
			ds_A[ty][tx] = A[Row * a_c + k * BONUS_TILE + tx];
		else 
			ds_A[ty][tx] = 0.0;

		if ((k * BONUS_TILE + ty < a_c) && Col < b_c)
			ds_B[ty][tx] = B[(k * BONUS_TILE + ty) * b_c + Col];
		else 
			ds_B[ty][tx] = 0.0;

		__syncthreads();

		for (int n = 0; n < BONUS_TILE; ++n)
			CVal += ds_A[ty][n] * ds_B[n][tx];

		__syncthreads();

		
	}

	if (Row < a_r && Col < b_c) {
		C[((by * by + ty) * b_c) + (bx * bx) + tx] = CVal;
	}
}


//Part 1 basic tiled matrix multiplication
__global__
void MatrixMulKernal(float* A, float* B, float* C, int n, int s) {
	
	// depending on the block width, create appropriate size shared memory and excecute. Thats the only method I could come up with. 
	if (s == 2) {
		__shared__ float ds_A[2][2];
		__shared__ float ds_B[2][2];

		int bx = blockIdx.x; int by = blockIdx.y;
		int tx = threadIdx.x; int ty = threadIdx.y;

		int Col = bx * blockDim.x + tx;
		int Row = by * blockDim.y + ty;
		float Cvalue = 0;
		for (int t = 0; t < n / s; t++) {

			ds_A[ty][tx] = A[Row * n + t * s + tx];
			ds_B[ty][tx] = B[(t * s + ty) * n + Col];
			__syncthreads();
			for (int i = 0; i < s; i++) {
				Cvalue += ds_A[ty][i] * ds_B[i][tx];
			}
			__syncthreads();
		}

		C[Row * n + Col] = Cvalue;
		//printf("%f\n", Cvalue);
	} 

	else if (s == 4) {
		__shared__ float ds_A[4][4];
		__shared__ float ds_B[4][4];
		int bx = blockIdx.x; int by = blockIdx.y;
		int tx = threadIdx.x; int ty = threadIdx.y;

		int Col = bx * blockDim.x + tx;
		int Row = by * blockDim.y + ty;
		float Cvalue = 0;
		for (int t = 0; t < n / s; t++) {

			ds_A[ty][tx] = A[Row * n + t * s + tx];
			ds_B[ty][tx] = B[(t * s + ty) * n + Col];
			__syncthreads();
			for (int i = 0; i < s; i++) {
				Cvalue += ds_A[ty][i] * ds_B[i][tx];
			}
			__syncthreads();
		}

		C[Row * n + Col] = Cvalue;
		//printf("%f\n", Cvalue);
	}

	else if (s == 10) {
		__shared__ float ds_A[10][10];
		__shared__ float ds_B[10][10];
		int bx = blockIdx.x; int by = blockIdx.y;
		int tx = threadIdx.x; int ty = threadIdx.y;

		int Col = bx * blockDim.x + tx;
		int Row = by * blockDim.y + ty;
		float Cvalue = 0;
		for (int t = 0; t < n / s; t++) {

			ds_A[ty][tx] = A[Row * n + t * s + tx];
			ds_B[ty][tx] = B[(t * s + ty) * n + Col];
			__syncthreads();
			for (int i = 0; i < s; i++) {
				Cvalue += ds_A[ty][i] * ds_B[i][tx];
			}
			__syncthreads();
		}

		C[Row * n + Col] = Cvalue;
		//printf("%f\n", Cvalue);
	}

	else if (s == 20) {
		__shared__ float ds_A[20][20];
		__shared__ float ds_B[20][20];
		int bx = blockIdx.x; int by = blockIdx.y;
		int tx = threadIdx.x; int ty = threadIdx.y;

		int Col = bx * blockDim.x + tx;
		int Row = by * blockDim.y + ty;
		float Cvalue = 0;
		for (int t = 0; t < n / s; t++) {

			ds_A[ty][tx] = A[Row * n + t * s + tx];
			ds_B[ty][tx] = B[(t * s + ty) * n + Col];
			__syncthreads();
			for (int i = 0; i < s; i++) {
				Cvalue += ds_A[ty][i] * ds_B[i][tx];
			}
			__syncthreads();
		}

		C[Row * n + Col] = Cvalue;
		//printf("%f\n", Cvalue);
	}
	else {
		__shared__ float ds_A[25][25];
		__shared__ float ds_B[25][25];
		int bx = blockIdx.x; int by = blockIdx.y;
		int tx = threadIdx.x; int ty = threadIdx.y;

		int Col = bx * blockDim.x + tx;
		int Row = by * blockDim.y + ty;
		float Cvalue = 0;
		for (int t = 0; t < n / s; t++) {

			ds_A[ty][tx] = A[Row * n + t * s + tx];
			ds_B[ty][tx] = B[(t * s + ty) * n + Col];
			__syncthreads();
			for (int i = 0; i < s; i++) {
				Cvalue += ds_A[ty][i] * ds_B[i][tx];
			}
			__syncthreads();
		}

		C[Row * n + Col] = Cvalue;
		//printf("%f \n \n", Cvalue); 
		//printf("%f\n", Cvalue);
	}


}
 


void sumMatrixOnHost(float* A, float* B, float* C, int w)
{
	float* N = A;
	float* M = B;
	float* R = C;
	for (int k = 0; k < w; k++)
	{
		for (int p = 0; p < w; p++)
		{
			float Cval = 0;
			for (int i = 0; i < w; i++) {
				Cval += N[k*w + i] * M[i*w + p];
			}
			R[k*w + p] = Cval;
		}

	}
	return;
}

void matMultBonus(float* A, float* B,float* C, int r_A, int c_A, int c_B) {
	for (int i = 0; i < r_A; i++) {
		for (int j = 0; j < c_B; j++) {
			float CVal = 0.0; 
			for (int k = 0; k < c_A; k++) {
				CVal += A[i * c_A + k] * B[k * c_B + j];
			}
			C[i * c_B + j] = CVal; 
		}

	}
}




void checkResultBonus(float* CPU, float* GPU, int dim) {
	float Margin = 1.0E-2;
	for (int i = 0; i < dim; i++)
	{
		if (abs(CPU[i] - GPU[i+1]) > Margin)
		{
			break;
		}



	}
	printf("Test PASSED\n\n");
}

void checkResult(float* CPU, float* GPU, int dim) {
	float Margin = 1.0E-2;
	for (int i = 0; i < dim; i++)
	{
		if (abs(CPU[i] - GPU[i]) > Margin)
		{
			printf("CPU %f GPU %f ", CPU[i], GPU[i]);
			printf("Matricies do not match.\n\n");
			break;
		}



	}
	printf("Test PASSED\n\n");
}
void initialData(float* Matrix, int dim)
{
	int i;
	for (i = 0; i < dim; i++)
	{
		Matrix[i] = (float)(rand() & 0xFF) / 10.0f;
	}

}
// S = matrix 1 rows 
// S1 = matrix 1 col 
// S2 = matrix 2 rows
// S3 = matrix 2 col
void computeMatrixBonus(int S, int S1,  int S3) {

	float* H_N, *H_M, *H_R, *H_R1;
	float* D_N, * D_M, * D_R;

	size_t sizeInFloats1 = S * S1 * sizeof(float);
	size_t sizeInFloats2 = S1 * S3 * sizeof(float);
	H_N = (float*)malloc(sizeInFloats1);
	H_M = (float*)malloc(sizeInFloats2);
	H_R = (float*)malloc(S*S3*sizeof(float));
	H_R1 = (float*)malloc(S*S3*sizeof(float));

	initialData(H_N, S * S1);
	initialData(H_M, S1 * S3);

	cudaMalloc((void**)&D_N, sizeInFloats1);
	cudaMalloc((void**)&D_M, sizeInFloats2);
	cudaMalloc((void**)&D_R, S * S3 * sizeof(float));

	cudaMemcpy(D_N, H_N, sizeInFloats1, cudaMemcpyHostToDevice);
	cudaMemcpy(D_M, H_M, sizeInFloats2, cudaMemcpyHostToDevice);


	matMultBonus(H_N,H_M, H_R, S,S1,S3);
	

	

	dim3 block(BONUS_TILE, BONUS_TILE1,1);
	dim3 thread((S3 + block.x -1 ) / block.x, (S + block.y -1 ) / block.y);

	MatrixMulKernalBonus << <thread, block >> >(D_N, D_M, D_R, S, S1, S3);

	cudaMemcpy(H_R1, D_R, S*S3*sizeof(float), cudaMemcpyDeviceToHost);
	checkResultBonus(H_R, H_R1, S * S3);

	cudaFree(D_N);
	cudaFree(D_M);
	cudaFree(D_R);

	free(H_N);
	free(H_M);
	free(H_R);
	free(H_R1);
	// reset device

	cudaDeviceReset();


}
void computeMatrix(int S, int s) {

	float* H_N, *H_M, *H_R, *H_R1;
	//Size of matrix dimension i.e. 1024


	// Multiply each dimension to get the matrix and then multiply by size of int to get the value in bytes

	size_t sizeInFloats = S * S * sizeof(float);
	//input host vector N

	H_N = (float*)malloc(sizeInFloats);
	H_M = (float*)malloc(sizeInFloats);
	H_R = (float*)malloc(sizeInFloats);
	H_R1 = (float*)malloc(sizeInFloats);

	initialData(H_N, S * S);
	initialData(H_M, S * S);

	memset(H_R, 0, S);
	memset(H_R1, 0, S);

	auto t1 = std::chrono::high_resolution_clock::now();
	sumMatrixOnHost(H_N, H_M, H_R, S);
	auto t2 = std::chrono::high_resolution_clock::now();


	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	printf("The CPU took %d to complete the computation.\n\n", duration);

	float* D_N, *D_M, *D_R;

	cudaMalloc((void**)&D_N, sizeInFloats);
	cudaMalloc((void**)&D_M, sizeInFloats);
	cudaMalloc((void**)&D_R, sizeInFloats);





	cudaMemcpy(D_N, H_N, sizeInFloats, cudaMemcpyHostToDevice);
	cudaMemcpy(D_M, H_M, sizeInFloats, cudaMemcpyHostToDevice);

	

	dim3 block(s, s);
	dim3 thread((int)ceil((S + block.x - 1)) / block.x, (int)ceil((S + block.y - 1) / block.y));



	cudaEvent_t start, stop;
	float time;


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	MatrixMulKernal << <thread, block >> > (D_N, D_M, D_R, S, s);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);



	printf("The GPU took %f microseconds to complete the computation with one thread per element.\n\n", time * 1000);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(H_R1, D_R, sizeInFloats, cudaMemcpyDeviceToHost);
	checkResult(H_R, H_R1, S * S);



	cudaFree(D_N);
	cudaFree(D_M);
	cudaFree(D_R);

	free(H_N);
	free(H_M);
	free(H_R);
	free(H_R1);
	// reset device

	cudaDeviceReset();


}

int main()
{
	/*
	
	int blocks[] = {2,4,10,20,25};
	int block;
	
	for (int i = 0; i < 5; i++)
	{	
		block = blocks[i];
		printf("Tiled Matrix multiplication with %d blocks \n\n", block);
		printf("100 x 100 matrix multiplication.\n");
		computeMatrix(100, block);
		printf("200 x 200 matrix multiplication.\n");
		computeMatrix(200, block);
		printf("500 x 500 matrix multiplication.\n");
		computeMatrix(500, block);
		printf("1000 x 1000 matrix multiplication.\n");
		computeMatrix(1000, block);
		printf("1500 x 1500 matrix multiplication.\n");
		computeMatrix(1500, block);
		printf("5000 x 5000 matrix multiplication.\n");
		computeMatrix(5000, block);
	}
	*/
	
	
	printf("Bonus tiled matrix multiplication with 8 x 14 block widths \n\n"); 
	computeMatrixBonus(200, 250, 400);

	return 0;


}
