#include <iostream>
#include <vector>
#include <fstream>
#include <stdint.h>
#include "reader.h"
#include "matrix.h"
#include "multiplication.h"

int N;
int rows;
int cols;

int main(int argc, char* argv[]) {
    Reader reader(argv[1], argv[2]);
    initialize(reader.vertices_count);

    std::vector<Matrix> matrices;
    matrices.reserve(static_cast<unsigned long>(reader.nonterm_count));

    for (int i = 0; i < reader.nonterm_count; ++i) {
        matrices.emplace_back();
    }

    for (auto& edge : reader.edges) {
        for (int nonterm : reader.term_to_nonterm[edge.first]) {
            matrices[nonterm].put_bit(edge.second.first, edge.second.second);
            matrices[nonterm].is_changed = true;
        }
    }

    start_time();

    for (int i = 0; i < reader.nonterm_count; ++i) {
        matrices[i].transfer_to_gpu();
    }

    wait_();

    while (true) {
        bool has_changed_global = false;
        for (auto& prod : reader.nonterm_prods) {
            if (matrices[prod.second.first].is_changed | matrices[prod.second.second].is_changed) {
                bool has_changed = 
                                MatrixMulAdd(matrices[prod.second.first].matrix_device, 
                                            matrices[prod.second.second].matrix_device, 
                                            matrices[prod.first].matrix_device);
                matrices[prod.first].is_changed = has_changed;
                has_changed_global |= has_changed;
            }
        }
        if (!has_changed_global) {
            break;
        }
    }

    for (int i = 0; i < reader.nonterm_count; ++i) {
        matrices[i].transfer_to_cpu();
    }

    wait_();

    stop_time();

    auto out_stream = std::ofstream(argv[3], std::ofstream::out);
    for (auto& nonterm : reader.nonterm_to_int) {
        out_stream << nonterm.first;
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                uint32_t cur_val = matrices[nonterm.second].matrix_host[r * cols + (c / 32)];
                if (cur_val & (1 << (31 - (c % 32)))) {
                    out_stream << " " << r << " " << c;
                }
            }
        }
        out_stream << std::endl;
    }
    out_stream.close();

return 0;
}
