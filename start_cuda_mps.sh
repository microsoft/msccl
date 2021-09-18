for x in 0 1 2 3
do
  nvidia-smi -i ${x} -c EXCLUSIVE_PROCESS
done
nvidia-cuda-mps-control -d
