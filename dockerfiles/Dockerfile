FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}


##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update && \
    apt-get install -y --allow-change-held-packages --no-install-recommends \
    software-properties-common \
    build-essential autotools-dev cmake g++ gcc \
    openssh-client openssh-server \
    nfs-common pdsh curl sudo net-tools \
    vim iputils-ping wget perl unzip

##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version

##############################################################################
# Pip
##############################################################################
# pip version <= 20.1.1 is needed for the ruamel.yaml installation conflict
# between conda and pip. ruamel.yaml is needed by azureml.
# https://github.com/Azure/MachineLearningNotebooks/issues/1110 for more info.
ENV PIP_VERSION=20.1.1
RUN conda install -y pip=${PIP_VERSION} && \
    # Print python an pip version
    python -V && pip -V

##############################################################################
# MPI
##############################################################################
RUN cd ${STAGE_DIR} && mkdir openmpi/ && cd openmpi && wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.1.tar.gz && \
    tar zxf openmpi-4.0.1.tar.gz && \
    cd openmpi-4.0.1 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf ${STAGE_DIR}/openmpi/

##############################################################################
# SCCL
##############################################################################

# update NCCL in pytorch, install SCCL interpreter
RUN pip uninstall torch -y

RUN pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

RUN conda install -c pytorch magma-cuda111 -y

ENV CMAKE_PREFIX_PATH=/opt/conda

# Change NCCL to SCCL Runtime
RUN cd ${STAGE_DIR} && \
    git clone https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git checkout tags/v1.9.0 -b v1.9.0_sccl && \
    perl -p -i -e  's/url = https:\/\/github\.com\/NVIDIA\/nccl/url = https:\/\/github\.com\/microsoft\/msccl/g' .gitmodules && \
    git submodule sync third_party/nccl  && \
    git submodule update --init --recursive  && \
    git submodule update --init --recursive --remote third_party/nccl && \
    cd third_party/nccl/nccl/ && \
    git checkout master && \
    cd ../../../ && \
    git apply third_party/nccl/nccl/patches/nccl.cpp.patch && \
    python setup.py install && \
    cd ${STAGE_DIR} && \
    rm -rf ${STAGE_DIR}/pytorch

# Install SCCL
RUN cd ${STAGE_DIR}/ && \
    git clone https://github.com/microsoft/sccl.git && \
    cd sccl/ && python setup.py install && \
    cd ${STAGE_DIR} && \
    rm -rf ${STAGE_DIR}/sccl/

##############################################################################
# inspector-topo
##############################################################################

RUN apt-get install libibverbs-dev libnuma-dev -y
RUN cd ${STAGE_DIR}/ && git clone https://github.com/microsoft/inspector-topo.git && \
    cd inspector-topo/ && make && make install
