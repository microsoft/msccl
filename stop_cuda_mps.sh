for x in 0 1 2 3
do
  nvidia-smi -i ${x} -c DEFAULT
done
echo quit | nvidia-cuda-mps-control
