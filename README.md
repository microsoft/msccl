# MSCCL

Microsoft Collective Communication Library (MSCCL) is a platform to execute custom collective communication algorithms for multiple accelerators supported by Microsoft Azure.

## Introduction

MSCCL is an inter-accelerator communication framework that is built on top of [NCCL](https://github.com/nvidia/nccl) and uses its building blocks to execute custom-written collective communication algorithms. MSCCL vision is to provide a unified, efficient, and scalable framework for executing collective communication algorithms across multiple accelerators. To achieve this, MSCCL has multiple capabilities:

- Programmibility: Inter-connection among accelerators have different latencies and bandwidths. Therefore, a generic collective communication algorithm does not necessarily well for all topologies and buffer sizes. MSCCL allows a user to write a hyper-optimized collective communication algorithm for a given topology and a buffer size. This is possbile through two main components: [MSCCL toolkit](https://github.com/microsoft/msccl-tools) and [MSCCL runtime](https://github.com/microsoft/msccl) (this repo). MSCCL toolkit contains a high-level DSL (MSCCLang) and a compiler which generate an IR for the MSCCL runtime (this repo) to run on the backend. MSCCL will always MSCCL will automatically fall back to a NCCL's generic algorithm in case there is no custom algorithm. [Example](#Example) provides some instances on how MSCCL toolkit with the runtime works. Please refer to [MSCCL toolkit](https://github.com/microsoft/msccl-tools) for more information.
- Profiling: MSCCL has a profiling tool [NPKit](https://github.com/microsoft/npkit) which provides detailed timeline for each primitive send and receive operation to understand the bottlenecks in a given collective communication algorithms.

MSCCL is the product of many great researchers and interns at Microsoft Research. Below is a list of our publications:

- [GC3: An Optimizing Compiler for GPU Collective Communication](https://arxiv.org/abs/2201.11840) -- Under submission
- [Synthesizing optimal collective algorithms](https://dl.acm.org/doi/10.1145/3437801.3441620) -- PPoPP'21 (Best Paper Award)
- [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads](https://arxiv.org/abs/2105.05720) -- ASPLOS'22
- [Synthesizing Collective Communication Algorithms for Heterogeneous Networks with TACCL](https://arxiv.org/abs/2111.04867) -- Under submission

Please consider citing our work if you use MSCCL in your research. Also, please contact us if you have any questions or need an optimized collective communication algorithm for a specific topology.

## Example

In order to use MSCCL customized algorithms, you may follow these steps to use two different MSCCL algorithms for AllReduce on Azure NDv4 which has 8xA100 GPUs:

Steps to install MSCCL:

```shell
$ git clone https://github.com/microsoft/msccl.git
$ cd msccl/
$ make -j src.build
$ cd ../
```

Then, follow these steps to install [nccl-tests](https://github.com/nvidia/nccl-tests) for performance evaluation:

```shell
$ git clone https://github.com/nvidia/nccl-tests.git
$ cd nccl-tests/
$ make MPI=1 NCCL_HOME=../msccl/build/ -j 
$ cd ../
```

To evaluate the performance, execute the following command line on an Azure NDv4 node:

```shell
$ mpirun -np 8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0  -x LD_LIBRARY_PATH=msccl/build/lib/:$LD_LIBRARY_PATH -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,ENV -x NCCL_NET_SHARED_BUFFERS=0 -x MSCCL_XML_FILES=allpairs.xml -x NCCL_ALGO=MSCCL,RING,TREE -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  nccl-tests/build/all_reduce_perf -b 128 -e 64MB -f 2 -g 1 -c 1 -n 1000 -w 1000 -z 0
```

If everything is installed correctly, you should see the following output in log:

```shell
[0] NCCL INFO Connected 2 MSCCL algorithms
```

The two algorithms are the two XML files specified by `MSCCL_XML_FILES` in the command line. You may remove `MSCCL` among `NCCL_ALGO` algorithms to disable the custom algorithms

## Build

To build the library:

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
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
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

Tests for MSCCL are maintained separately at https://github.com/nvidia/nccl-tests.

```shell
$ git clone https://github.com/nvidia/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.

All modifications are copyright (c) 2020-2022, Microsoft Corporation. All rights reserved.
