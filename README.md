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
- [TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches](https://arxiv.org/abs/2111.04867) -- NSDI'23

Please consider citing our work if you use MSCCL in your research. Also, please contact us if you have any questions or need an optimized collective communication algorithm for a specific topology.

## Example

In order to use MSCCL customized algorithms, you may follow these steps to use two different MSCCL algorithms for AllReduce on Azure NDv4 which has 8xA100 GPUs:

Steps to install MSCCL:

```sh
$ git clone https://github.com/microsoft/msccl.git
$ cd msccl/
$ make -j src.build
$ cd ../
```

Then, follow these steps to install [nccl-tests](https://github.com/nvidia/nccl-tests) for performance evaluation:

```sh
$ git clone https://github.com/nvidia/nccl-tests.git
$ cd nccl-tests/
$ make MPI=1 NCCL_HOME=../msccl/build/ -j 
$ cd ../
```

Next install [MSCCL toolkit](https://github.com/microsoft/msccl-tools) to compile a few custom algorithms:

```sh
$ git clone https://github.com/microsoft/msccl-tools.git
$ cd msccl-tools/
$ pip install .
$ cd ../
$ python msccl-tools/examples/mscclang/allreduce_a100_allpairs.py --protocol=LL 8 2 > test.xml
$ cd ../
```

The compiler's generated code is an XML file (`test.xml`) that is fed to MSCCL runtime. To evaluate its performance, execute the following command line on an Azure NDv4 node or any 8xA100 system:

```sh
$ mpirun -np 8 -x LD_LIBRARY_PATH=msccl/build/lib/:$LD_LIBRARY_PATH -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,ENV -x MSCCL_XML_FILES=test.xml -x NCCL_ALGO=MSCCL,RING,TREE  nccl-tests/build/all_reduce_perf -b 128 -e 32MB -f 2 -g 1 -c 1 -n 100 -w 100 -G 100 -z 0
```

If everything is installed correctly, you should see the following output in log:

```sh
[0] NCCL INFO Connected 1 MSCCL algorithms
```

`test.xml` is passed in to the runtime by `MSCCL_XML_FILES` in the command line. You may evaluate the performance of `test.xml` by comparing in-place (the new algorithm) vs out-of-place (default ring algorithm) and it should up-to 2-3x faster on 8xA100 NVLink-interconnected GPUs. [MSCCL toolkit](https://github.com/microsoft/msccl-tools) has a rich set of algorithms for different Azure SKUs and collective operations with significant speedups over vanilla NCCL.

## Build

To build the library:

```sh
$ cd msccl
$ make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```sh
$ make src.build CUDA_HOME=<path to cuda install>
```

MSCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, MSCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :
```sh
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

## Install

To install MSCCL on the system, create a package then install it as root.

Debian/Ubuntu :
```sh
$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper fakeroot
$ # Build NCCL deb package
$ make pkg.debian.build
$ ls build/pkg/deb/
```

RedHat/CentOS :
```sh
$ # Install tools to create rpm packages
$ sudo yum install rpm-build rpmdevtools
$ # Build NCCL rpm package
$ make pkg.redhat.build
$ ls build/pkg/rpm/
```

OS-agnostic tarball :
```sh
$ make pkg.txz.build
$ ls build/pkg/txz/
```

## PyTorch Integration

For integration with PyTorch, follow the dockerfile in this repo. It has an example for how to replace default NCCL with MSCCL.

## NPKit Integration

MSCCL integrates [NPKit](https://github.com/microsoft/npkit), a profiler framework that enables collecting fine-grained trace events in MSCCL components that identifies transmission bottlenecks.

To Enable NPKit, simply add `NPKIT=1` along with your make command. During execution, environment variable `NPKIT_DUMP_DIR` will be used to produce all of the output (one output file per rank). By default, `/tmp/` will be used.

To analyze NPKit output, run python script `tools/npkit_trace_generator.py` to get the final `.json` file which can be viewed by a trace viewer such as Microsoft Edge `edge://tracing` or Google Chrome `chrome://tracing`.

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.

All modifications are copyright (c) 2020-2022, Microsoft Corporation. All rights reserved.
