mpirun \
    --allow-run-as-root --tag-output \
    -map-by ppr:4:node --bind-to numa \
    -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -x NCCL_ALGO=Ring \
    -x NCCL_PROTO=^LL \
    -x NCCL_GRAPH_FILE=./graph.xml \
    /nccl-tests/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -c 1 -w 200 -n 50
