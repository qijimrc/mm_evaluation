#! /bin/bash
WORLD_SIZE=${SLURM_NTASKS:-1}
RANK=${SLURM_PROCID:-0}
# MASTER_ADDR is the first in SLURM_NODELIST
if [ -z "$SLURM_NODELIST" ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=27878
else
    MASTER_ADDR=`scontrol show hostnames $SLURM_NODELIST | head -n 1`
    MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
fi
# generate a port at random
LOCAL_RANK=${SLURM_LOCALID:-0}
MP_SIZE=1

# nccl enviroments
server_addr=$(cat .server)
echo "server addr: $server_addr"

if [ "$server_addr" == "zhongwei" ] || [ "$server_addr" == "zhongwei2" ]; then
    # zhongwei: https://lslfd0slxc.feishu.cn/wiki/Zr8cwxiGyixISQkwxo6cjHQqngg
    # zhongwei2: https://lslfd0slxc.feishu.cn/wiki/A2DswoQ0wiYpv2kqt9IcTKPhn5f
    echo "it is in $server_addr"
    module unload cuda
    module load cuda/11.8
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=mlx5_0:1,mslx5_1:1,mlx5_4:1,mlx5_5:1   
    export NCCL_SOCKET_IFNAME=bond0 
    export NCCL_DEBUG=INFO
elif [ "$server_addr" == "wulan" ]; then
    # wulan: https://lslfd0slxc.feishu.cn/wiki/FmSqwOUtSi3VXrkUUR3clJZMnxh
    echo "it is in wulan"
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export NCCL_PXN_DISABLE=0
    export NCCL_IB_GID_INDEX=3
    export NCCL_NVLS_ENABLE=1
    export NCCL_IB_TIMEOUT=22
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8
    export NCCL_IB_TC=136
    export NCCL_IB_QPS_PER_CONNECTION=8
    export NCCL_IB_SL=5
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=bond0
    export NCCL_DEBUG=INFO
elif [ "$server_addr" == "jinan" ]; then
    # jinan: https://lslfd0slxc.feishu.cn/wiki/HAitww42Ei6uImkBc2Qc29Ixnjb
    echo "it is in jinan"
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=2
else
    echo "unknown server addr: $server_addr"
fi
echo "RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

HOST_FILE_PATH="hostfile"

# eval_tasks="TouchStoneChinese TouchStone Infer"
eval_models="YiVL_6B"
# VQAv2 TextVQA TDIUC ScienceQA OKVQA OCRVQA 
eval_tasks="NoCaps COCO GQA OCR_EN OCR_ZH STVQA FlickrCap ChartQA DocVQA VizWizVQA TallyQA MMMU MMBench"

gpt_options=" \
       --batch_size 1 \
       --eval_tasks $eval_tasks \
       --eval_models $eval_models \
       --log_interval 10 \
       --pad_noimg \
       --use_debug_mode 1000 \
"

run_cmd="SERVER_ADDR=$server_addr WORLD_SIZE=$WORLD_SIZE RANK=$RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT LOCAL_RANK=$LOCAL_RANK LOCAL_WORLD_SIZE=8 python3 example/evaluate.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
echo "DONE on `hostname`"