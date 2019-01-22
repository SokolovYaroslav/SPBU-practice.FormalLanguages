#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <ctime>
#include "multiplication.h"

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
		{
			fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define BITS_PER_TYPE 32
#define THREADS_PER_BLOCK 32 // More is faster if N > 5000

__device__ bool is_changed;

size_t matrix_memsize;
uint32_t *tmp_matrix;
clock_t begin;
clock_t end;

void initialize(int N_inp) {
	int devID = 0;
	devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));

	N = N_inp;
	rows = N;
	cols = N / BITS_PER_TYPE + (N % BITS_PER_TYPE ? 1 : 0);
	matrix_memsize = rows * cols * sizeof(uint32_t);

	gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&tmp_matrix), matrix_memsize));
}

__global__ void DummyMulAdd(uint32_t *A, uint32_t *B, uint32_t *C, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;

    if (x >= cols) {
        return;
    }

	uint32_t acc = 0;
	uint32_t a_el;
	for (uint32_t i = 0; i < cols; ++i) {
		a_el = A[y * cols + i];
		#pragma unroll
		for (uint32_t b = 0; b < 32; ++b) {
			if (a_el & 1) {
				acc |= B[x + 32 * cols * i + cols * (31 - b)];
			}
			a_el >>= 1;
		}
	}

	if (acc == 0) {
		return;
	}

	uint32_t c_old = C[y * cols + x];
	if (c_old != (acc | c_old)) {
		is_changed = true;
		C[y * cols + x] = acc | c_old;
	}
}

__global__ void DummyMul(uint32_t *A, uint32_t *B, uint32_t *C, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;

    if (x >= cols) {
        return;
    }

	uint32_t acc = 0;
	uint32_t a_el;
	for (uint32_t i = 0; i < cols; ++i) {
		a_el = A[y * cols + i];
		#pragma unroll
		for (uint32_t b = 0; b < 32; ++b) {
			if (a_el & 1) {
				acc |= B[x + 32 * cols * i + cols * (31 - b)];
			}
			a_el >>= 1;
		}
	}

	C[y * cols + x] = acc;
}


__global__ void AddToLeft(uint32_t *A, uint32_t *B, int cols) {
	int index = blockIdx.y * cols + blockIdx.x * blockDim.x + threadIdx.x;

    if ((blockIdx.x * blockDim.x + threadIdx.x) >= cols) {
        return;
    }

	uint32_t A_el = A[index];
	uint32_t res = B[index] | A_el;
	if (res != A_el) {
		is_changed = true;
		A[index] = res;
	}
}

void wait_() {
	cudaDeviceSynchronize();
}

void start_time() {
	begin = clock();
}

void stop_time() {
	end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("%d\n", (int) (elapsed_secs * 1000 + 0.5));
}

void gpuMemset(uint32_t *d_M, int val) {
	gpuErrchk(cudaMemset(d_M, val, matrix_memsize));
}

uint32_t * device_matrix_alloc() {
	uint32_t *d_M;
	gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_M), matrix_memsize));

	return d_M;
}

uint32_t * host_matrix_calloc() {
    uint32_t *M;
    gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&M), matrix_memsize));
    gpuMemset(M, 0);
	return M;
}

void gpu2cpu_async(uint32_t *d_M, uint32_t *h_M) {
	gpuErrchk(cudaMemcpyAsync(h_M, d_M, matrix_memsize, cudaMemcpyDeviceToHost));
}

void cpu2gpu_async(uint32_t *h_M, uint32_t *d_M) {
	gpuErrchk(cudaMemcpyAsync(d_M, h_M, matrix_memsize, cudaMemcpyHostToDevice));
}

void setFlag() {
	bool flag = false;
	gpuErrchk(cudaMemcpyToSymbol(is_changed, &flag, sizeof(bool)))
}

uint8_t getFlag() {
	bool flag;
	gpuErrchk(cudaMemcpyFromSymbol(&flag, is_changed, sizeof(bool)))

	return flag;
}

bool MatrixMulAdd(uint32_t *A, uint32_t *B, uint32_t *C) {
	bool safe = (A == C) || (B == C);
	dim3 mul_threads(THREADS_PER_BLOCK);
	dim3 mul_blocks(cols / THREADS_PER_BLOCK + (cols % THREADS_PER_BLOCK ? 1 : 0), rows);

    setFlag();
	if (safe) {
		DummyMul <<<mul_blocks, mul_threads>>> (A, B, tmp_matrix, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
		AddToLeft <<<mul_blocks, mul_threads>>> (C, tmp_matrix, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
	}
	else {
		DummyMulAdd <<<mul_blocks, mul_threads>>> (A, B, C, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
	}

	return getFlag();
}
