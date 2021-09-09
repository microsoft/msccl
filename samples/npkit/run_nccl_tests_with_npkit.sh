set -x

function nccl_test() {
  mpirun \
    --allow-run-as-root --tag-output \
    -hostfile $5 -map-by ppr:8:node --bind-to numa \
    -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 \
    -x PATH \
    -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so \
    -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/:$LD_LIBRARY_PATH \
    -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -x UCX_NET_DEVICES=eth0 \
    -x UCX_TLS=tcp \
    -x UCX_IB_ENABLE_CUDA_AFFINITY=n \
    -x UCX_IB_PCI_RELAXED_ORDERING=on \
    -x UCX_IB_IS_GLOBAL=1 \
    -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
    -x NCCL_NET_GDR_LEVEL=5 \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_TOPO_FILE=/opt2/msft/topo.xml \
    -x NCCL_PLUGIN_P2P=ib \
    -x NCCL_ALGO=$2 \
    -x NCCL_PROTO=$3 \
    -x NCCL_IB_IS_GLOBAL=1 \
    -x NPKIT_DUMP_DIR=$4 \
    /nccl-tests/build/all_reduce_perf -b $1 -e $1 -f 2 -g 1 -c 0 -w $6 -n $7 > $4/log.txt
}

num_warmups=200
num_iters=50
num_kernel_runs=$(((${num_warmups}+${num_iters}+1)*2))
num_kernel_runs_to_sample=10
data_size="16M"
nccl_algo="Ring"
nccl_proto="LL128"
npkit_src_dir="/mnt/ziyyang/npkit_dev/nccl"
hostfile="${npkit_src_dir}/samples/npkit/hostfile"
npkit_run_dir="/mnt/ziyyang/npkit_run"
npkit_run_src_dir="${npkit_run_dir}/npkit_src"
npkit_dump_dir="${npkit_run_dir}/npkit_dump/${data_size}/${nccl_algo}/${nccl_proto}"
npkit_post_process_dir="${npkit_run_dir}/npkit_post_process/${data_size}/${nccl_algo}/${nccl_proto}"
npkit_result_dir="${npkit_run_dir}/npkit_result/${data_size}/${nccl_algo}/${nccl_proto}"
# npkit_flags=""
npkit_flags="-DENABLE_NPKIT -DENABLE_NPKIT_EVENT_NET_SEND_POSTED -DENABLE_NPKIT_EVENT_NET_SEND_DONE"
# npkit_flags="-DENABLE_NPKIT -DENABLE_NPKIT_EVENT_TIME_SYNC_CPU -DENABLE_NPKIT_EVENT_TIME_SYNC_GPU -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_ENTRY -DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_EXIT"

cd ${npkit_src_dir}
make clean
parallel-ssh -t 0 -h ${hostfile} "rm -rf ${npkit_run_dir}; mkdir -p ${npkit_dump_dir}"
parallel-scp -r -h ${hostfile} ${npkit_src_dir} ${npkit_run_src_dir}
parallel-ssh -t 0 -h ${hostfile} "cd ${npkit_run_src_dir}; make -j src.build NPKIT_FLAGS=\"${npkit_flags}\" NVCC_GENCODE=\"-gencode=arch=compute_80,code=sm_80\"; make install"

nccl_test ${data_size} ${nccl_algo} ${nccl_proto} ${npkit_dump_dir} ${hostfile} ${num_warmups} ${num_iters}
tail -n10 ${npkit_dump_dir}/log.txt
parallel-ssh -t 0 -h ${hostfile} "cd ${npkit_run_src_dir}/samples/npkit; python3 npkit_post_process.py --npkit_dump_dir=${npkit_dump_dir} --npkit_event_header_path=${npkit_run_src_dir}/src/include/npkit/npkit_event.h --output_dir=${npkit_post_process_dir} --num_kernel_runs=${num_kernel_runs} --num_kernel_runs_to_sample=${num_kernel_runs_to_sample}; cd ${npkit_post_process_dir}; tar cvzf npkit_result.tar.gz npkit_event_trace.json gpu_stage_durations.json cpu_stage_durations.json"
cat ${hostfile} | while read hostname
do
  mkdir -p ${npkit_result_dir}/${hostname}
  scp ${hostname}:${npkit_post_process_dir}/npkit_result.tar.gz ${npkit_result_dir}/${hostname}
done
cp ${npkit_dump_dir}/log.txt ${npkit_result_dir}
