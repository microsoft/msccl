# Example Usage of SHMCPU Proxy

1. Build nccl-tests with MPI.

2. Build and install NCCL with SHMCPU patch.

3. Compile `cuda_proxy.cc` with MPI.

        $ nvcc -O2 -lmpi -o cuda_proxy cuda_proxy.cc

4. Start CUDA MPS service. For each GPU, run:

        $ export CUDA_VISIBLE_DEVICES=<GPU-ID>
        $ nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
        $ nvidia-cuda-mps-control -d

    You can also reference `start_cuda_mps.sh`.

5. Start cuda_proxy processes for each GPU inside the machine with MPI. For example, if there are 4 GPUs in the machine, run:

        $ mpirun \
            --allow-run-as-root --tag-output \
            -map-by ppr:4:node --bind-to numa \
            -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
            ./cuda_proxy

    You can also reference `start_cuda_proxy.sh`.

6. Run nccl-tests. We use a pre-generated graph file because there are potential bug for graph search in NCCL 2.10.3. For performance test, switch `-c` off. 

        $ mpirun \
            --allow-run-as-root --tag-output \
            -map-by ppr:4:node --bind-to numa \
            -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
            -x NCCL_ALGO=Ring \
            -x NCCL_PROTO=^LL \
            -x NCCL_GRAPH_FILE=./graph.xml \
            /path/to/nccl-tests/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -c 1 -w 200 -n 50

    You can also reference `run_nccl_tests.sh`.

7. Stop CUDA MPS service.

        $ nvidia-smi -i 0 -c DEFAULT
        $ echo quit | nvidia-cuda-mps-control

    You can also reference `stop_cuda_mps.sh`.
