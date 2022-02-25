#include "devcomm.h"

__global__ void scclSynchronize(int workIndex, struct ncclDevComm* comm);