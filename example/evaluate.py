import os
import argparse

from mmdoctor.common.logger import log
from mmdoctor.evaluator import Evaluator

_server_addr = os.environ.get('SERVER_ADDR', 'zhongwei')

task_cache_dir = {
    "wulan": "/mnt/shared/img_datasets/mmbench_datasets/processed",
    "zhongwei": "/share/img_datasets/mmbench_datasets/processed_mmbench_20231120",
}[_server_addr]

model_cache_dir = {
    "zhongwei": "/share/official_pretrains/mm_evaluation"
}[_server_addr]

from mmdoctor.arguments import get_args
parser = argparse.ArgumentParser()

known, args_list = parser.parse_known_args()
args = get_args(args_list)

args.model_cache_dir = model_cache_dir
args.task_cache_dir = task_cache_dir

evaluator = Evaluator()

log(evaluator.get_task_names())
log(evaluator.get_metric_names())
log(evaluator.get_model_names())

# model_name = "XComposer"
# model_name = "XComposer2"
# model_name = "Emu2"
# model_name = "Emu2_Chat"
# model_name = "QwenVL"
# model_name = "QwenVLChat"
# model_name = "YiVL_6B"
# model_name = "YiVL_34B" OOM
# model_name = "QwenVLPlus"
# model_name = "QwenVLMax"
# model_name = "YiVL_6B"

for model_name in args.eval_models:
    log(f"Start evaluating {model_name}")
    evaluator.run(args, model_name)
    log("done")