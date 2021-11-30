# MSCCL

Microsoft Collective Communication Library (MSCCL) is a platform to execute custom collective communication algorithms for multiple accelerators supported by Microsoft Azure.

## Introduction

MSCCL is an inter-accelerator communication framework that is built on top of NCCL (https://github.com/nvidia/nccl) and uses its building blocks to execute custom-written collective communication algorithms. MSCCL vision is to provide a unified, efficient, and scalable framework for executing collective communication algorithms across multiple accelerators. To achieve this, MSCCL has multiple capabilities:

- Programmibility: Inter-connection among accelerators have different latencies and bandwidths. Therefore, a generic collective communication algorithm does not necessarily well for all topologies and buffer sizes. MSCCL vision is to provide a unique collective communication algorithm for each topology and buffer size. In case no algorithm is available, MSCCL will automatically fall back to a NCCL's generic algorithm. For information regarding how to select a specific algorithm, please refer to [SCCL](https://github.com/microsoft/sccl).
- Topology-Aware Drivers: each accelerator and type of inter-connect requires a driver to complete a primtive send and receive operation. MSCCL provides a link-specific driver for each Azure hardware to maximize the output of the connection.
- Profiling: MSCCL has a profiling tool which provides detailed timeline for each primitive send and receive operation to understand the bottlenecks in a given collective communication algorithms.

MSCCL is the product of many great researchers and interns at Microsoft Research. Below is a list of our publications:

- [Synthesizing optimal collective algorithms](https://dl.acm.org/doi/10.1145/3437801.3441620) -- PPoPP'21 (Best Paper Award)
- [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads](https://arxiv.org/abs/2105.05720) -- ASPLOS'22
- [Synthesizing Collective Communication Algorithms for Heterogeneous Networks with TACCL](https://arxiv.org/abs/2111.04867) -- Under Review

Please consider citing our work if you use MSCCL in your research. Also, please contact us if you have any questions or need an optimized collective communication algorithm for a specific topology.

## Build

To build the library :

```shell
$ cd msccl
$ make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```shell
$ make src.build CUDA_HOME=<path to cuda install>
```

MSCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, MSCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :
```shell
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
```

## Install

To install MSCCL on the system, create a package then install it as root.

Debian/Ubuntu :
```shell
$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper fakeroot
$ # Build NCCL deb package
$ make pkg.debian.build
$ ls build/pkg/deb/
```

RedHat/CentOS :
```shell
$ # Install tools to create rpm packages
$ sudo yum install rpm-build rpmdevtools
$ # Build NCCL rpm package
$ make pkg.redhat.build
$ ls build/pkg/rpm/
```

OS-agnostic tarball :
```shell
$ make pkg.txz.build
$ ls build/pkg/txz/
```

## Tests

Tests for MSCCL are maintained separately at https://github.com/parasailteam/nccl-tests.

```shell
$ git clone https://github.com/parasailteam/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
