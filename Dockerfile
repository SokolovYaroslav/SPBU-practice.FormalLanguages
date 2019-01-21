FROM nvidia/cuda:10.0-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.0 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get upgrade -y

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

RUN apt-get install -y cmake
