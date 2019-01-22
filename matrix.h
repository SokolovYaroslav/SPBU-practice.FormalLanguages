#pragma once

#include <stdint.h>
#include <iostream>
#include "multiplication.h"


struct Matrix {
    uint32_t *matrix_host;
    uint32_t *matrix_device;
    bool is_changed_prev;
    bool is_changed;

    Matrix() {
        is_changed = false;
        matrix_host = host_matrix_calloc();
        matrix_device = device_matrix_alloc();
    }

    ~Matrix() {
    }

    void transfer_to_gpu() {
        cpu2gpu(matrix_host, matrix_device);
    }

    void transfer_to_cpu() {
        gpu2cpu(matrix_device, matrix_host);
    }

    void put_bit(int i, int j) {
        matrix_host[i * cols + (j / 32)] |= 1U << (31 - (j % 32));
    }
};
