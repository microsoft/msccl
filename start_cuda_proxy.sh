mpirun \
    --allow-run-as-root --tag-output \
    -map-by ppr:4:node --bind-to numa \
    -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ./cuda_proxy
