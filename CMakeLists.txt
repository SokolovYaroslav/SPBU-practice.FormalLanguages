cmake_minimum_required(VERSION 3.10)
project(SEMEN CXX CUDA)

set(CMAKE_CXX_STANDART 11)

add_executable(prog multiplication.h multiplication.cu main.cpp reader.h matrix.h)