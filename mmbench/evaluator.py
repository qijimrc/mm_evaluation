import os
import sys
import copy
import torch
import logging
import argparse
import jsonlines

if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sat.helpers import print_rank0
from omegaconf import OmegaConf

from datetime import datetime
from mmbench.common.registry import Registry

class Evaluator:
    def __init__(self,
                 data_home_dir: str,
                 custom_cfg_path: str=None,
                 custom_functions: dict=dict(),
                 custom_dataset_functions: dict=dict(),
                 cfg_path: str=os.path.dirname(__file__)+'/config.yaml') -> None:
        """
        Args:
            data_home_dir (str): directory of dataset home
            custom_cfg_path (str, optional): _description_. Defaults to None.
            custom_functions (dict, optional): {"forward_step": func, ...} or {"task_name": {"forward_step": func, ...}, ...}
            custom_dataset_functions (dict, optional): {"normal_qa": func, ...} or {"task_name": {"normal_qa": func, ...}, ...}
            cfg_path (str, optional): _description_. Defaults to os.path.dirname(__file__)+'/config.yaml'.
        """
        self.default_cfg = OmegaConf.load(cfg_path)
        if custom_cfg_path:
            self._update_params(OmegaConf.load(custom_cfg_path))
        
        self.mmeval_home = os.environ.get("MMEVAL_HOME", os.path.join(os.path.expanduser('~'), ".mmbench_eval_tmp"))
        if not os.path.exists(self.mmeval_home):
            os.mkdir(self.mmeval_home)
            print_rank0(f"Using mmeval home: {self.mmeval_home}")
        self.data_home_dir = data_home_dir
        
        self.tasks = {
            name: Registry.get_task_class(name)(self.default_cfg["tasks"][level][name], \
                                                **{"custom_functions": self.get_custom_functions(custom_functions, name), \
                                                   "custom_dataset_functions": self.get_custom_functions(custom_dataset_functions, name)})
              for level in self.default_cfg["tasks"].keys() for name in self.default_cfg["tasks"][level]
        }
        
        self.eval_model_name = None
        self.eval_task_name = None
    
    def get_task_names(self):
        return Registry.list_tasks()
    
    def get_metric_names(self):
        return Registry.list_metrics()

    def get_model_names(self):
        return Registry.list_models()

    def get_custom_functions(self, custom_function_dict, task_name):
        ret = custom_function_dict[task_name] if task_name in custom_function_dict else {}
        task_names = set(self.get_task_names())
        for k,v in custom_function_dict.items():
            if k not in task_names:
                ret[k] = v
        return ret

    def print_info(self, print_str, add_sep_lines=False, sep_char='#', level=logging.INFO):
        print_items = [self.eval_model_name]
        if self.eval_task_name:
            print_items.append(self.eval_task_name)
        print_str = f"[{'-'.join(print_items)}] {print_str}"
        if add_sep_lines:
            print_str = f"{sep_char * 80}\n{print_str}\n{sep_char * 80}"
        print_rank0(print_str, level=level)
        
    def _update_params(self, custom_params):
        """update self.default_cfg using custom_params
        """
        if 'tasks' in custom_params:
            for level_name in custom_params['tasks'].keys():
                for task_name, params in custom_params['tasks'][level_name].items():
                    if task_name not in self.default_cfg["tasks"][level_name]:
                        self.default_cfg["tasks"][level_name][task_name] = {}
                    self.default_cfg["tasks"][level_name][task_name].update(params)

    def _evaluate_tasks(self, args, model_cls, eval_tasks=[]):
        model_scores = {}
        # start
        failed_tasks = []
        for i, task_name in enumerate(eval_tasks):
            try:
                self.eval_task_name = task_name
                c_task = self.tasks[task_name]
                self.print_info(f'Start ({i+1}/{len(eval_tasks)})')
                # reset args & model states
                args_cp = copy.deepcopy(args)
                args_cp.save_details_result_path = os.path.join(args_cp.save,
                                                                f'{self.eval_model_name}_{self.eval_task_name}')
                self.print_info(f"Detailed results will be saved in {args_cp.save_details_result_path}")
                # evaluate
                args_cp.eval_task_name = self.eval_task_name
                args_cp.eval_model_name = self.eval_model_name
                model_scores[self.eval_task_name] = c_task.do_evaluate(args_cp, model_cls)
                # save
                if args.rank == 0:
                    with jsonlines.open(args.save_result_path, mode='a') as fp:
                        _tmp = {"model": self.eval_model_name, "task": self.eval_task_name, "results": model_scores[task_name]}
                        fp.write(_tmp)
                self.print_info(f'Score: {model_scores[self.eval_task_name]}', add_sep_lines=True, sep_char='-')
                self.print_info(f'End ({i+1}/{len(eval_tasks)})')
            except Exception as e:
                import traceback
                self.print_info(traceback.format_exc(), level=logging.ERROR)
                failed_tasks.append(task_name)
            self.eval_task_name = None
        self.print_info(f'Complete.')
        if len(failed_tasks) > 0:
            self.print_info(f'Failed Tasks: {failed_tasks}', level=logging.WARNING)
        return model_scores

    def run(self, args, model_names, eval_tasks=[]):
        # update args
        args.rank = int(os.environ.get('RANK', 0))
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        timestring = datetime.now().strftime("%m-%d-%H-%M")
        args.data_home_dir = self.data_home_dir
        if not (hasattr(args, "save") and args.save):
            args.save = args.save if hasattr(args, 'save') and args.save else self.mmeval_home
        args.save = os.path.join(args.save, f'mmevaluation-{timestring}')
        args.save_result_path = f"{args.save}/scores.jsonl"
        print_rank0(f"All evaluate results will be saved in {args.save_result_path}")
        if not os.path.exists(args.save):
            print_rank0(f"Create save home: {args.save}")
            os.makedirs(args.save, exist_ok=True)
        # evaluate models
        all_scores = {}
        if not isinstance(model_names, list):
            model_names = [model_names]
        for i, model_cls in enumerate(model_names):
            if isinstance(model_cls, str):
                self.eval_model_name = model_cls
                model_cls = Registry.get_model_class(model_cls)(self.default_cfg['models'][model_cls], 
                                                                args)
            else:
                self.eval_model_name = model_cls.__class__.__name__
            model_cls.name = self.eval_model_name
            assert hasattr(model_cls, "generate"), f"Not exist `generate` method in {model_cls}!"
            # evaluate
            all_scores[self.eval_model_name] = self._evaluate_tasks(args, model_cls, eval_tasks)
            print_rank0(f'Model ({i+1}/{len(model_names)}) end: {self.eval_model_name}')
        # TODO: results merge
        print_rank0(f'Results are saved in {args.save_result_path}')
        print_rank0('DONE.')
        return all_scores

if __name__ == "__main__":
    data_home_dir = {
        "wulan": "/mnt/shared/img_datasets/mmbench_datasets/processed",
        "zhongwei": "/share/img_datasets/mmbench_datasets/processed_mmbench_20231120",
        "zhongwei2": "/zhipu-data/img_datasets/mmbench_datasets/processed_mmbench_20231120"
    }["zhongwei"]

    from mmbench.arguments import get_args
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_tasks', type=str, nargs='+', help='Specify the tasks for evaluation')
    known, args_list = parser.parse_known_args()
    args = get_args(args_list)
    
    args.model_cache_dir = "/share/official_pretrains/mm_evaluation"
    args.use_debug_mode = True
    # 

    # some hack code to run on multiple gpus
    import torch.distributed as dist
    torch.cuda.set_device(0)
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '7878')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://{}:{}'.format(master_addr, master_port),
        world_size=world_size,
        rank=rank
    )

    evaluator = Evaluator(data_home_dir=data_home_dir)
    
    print(evaluator.get_task_names())
    print(evaluator.get_metric_names())
    print(evaluator.get_model_names())
    
    # for model_name in evaluator.get_model_names():
    #     if model_name == "CogVLM":
    #         continue

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
    model_name = "CogVLM"
    
    if model_name == "CogVLM":
        args.model_cache_dir = "/share/official_pretrains/hf_home"
    
    print(f"Start evaluating {model_name}")
    evaluator.run(args, model_name, eval_tasks=["MMMU"])
    print("done")