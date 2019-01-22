#pragma once

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <string>

extern int N;
extern int rows;
extern int cols;

void initialize(int N_inp);

void wait_();

void start_time();

void stop_time();

uint32_t * device_matrix_alloc();

uint32_t * host_matrix_calloc();

void gpu2cpu_async(uint32_t *d_M, uint32_t *h_M);

void cpu2gpu_async(uint32_t *h_M, uint32_t *d_M);

bool MatrixMulAdd(uint32_t *A, uint32_t *B, uint32_t *C);