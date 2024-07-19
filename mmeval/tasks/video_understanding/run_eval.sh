#!/bin/bash
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
base_dir=$(dirname $(dirname $script_dir))

echo $base_dir

export PYTHONPATH=$base_dir

# Change CUDA toolkit
# export CUDA_HOME=/data/qiji/tools/cuda12-1/
# export PATH=$CUDA_HOME/bin:$PATH
# export CPATH=$CUDA_HOME/include:$CPATH
# export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# pip install -r $base_dir/requirements.txt

MODELPATH=/data/qiji/models/LLaVA-NeXT-Video-7B
SAVEDIR=/data/qiji/models/LLaVA-NeXT-Video-7B/output

# python $script_dir/eval_video_mme_qiji.py --model_path $MODELPATH --save_dir $SAVEDIR --num_process 8 --num_per_device 2
python $script_dir/eval_activitynet_qa.py --model_path $MODELPATH --save_dir $SAVEDIR --num_process 1 --num_per_device 1
# python $script_dir/eval_videochatgpt_qa.py --model_path $MODELPATH --save_dir $SAVEDIR --num_process 4 --num_per_device 2
# python $script_dir/eval_msvd_qa.py --model_path $MODELPATH --save_dir $SAVEDIR --num_process 4 --num_per_device 2
# python $script_dir/eval_msrvtt_qa.py --model_path $MODELPATH --save_dir $SAVEDIR --num_process 8 --num_per_device 3 
# python $script_dir/eval_mvbench.py --model_path $MODELPATH --save_dir $SAVEDIR --num_process 4 --num_per_device 2
# python $script_dir/eval_nextqa.py --model_path $MODELPATH --save_dir $SAVEDIR --num_process 1 --num_per_device 1
# python $script_dir/eval_tgif_qa.py --model_path $MODELPATH --save_dir $SAVEDIR --num_process 16 --num_per_device 2