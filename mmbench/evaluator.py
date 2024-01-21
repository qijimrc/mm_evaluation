import os
import sys
import copy
import torch
import logging
import argparse
import jsonlines

if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sat import mpu
from sat.helpers import print_rank0
from omegaconf import OmegaConf

from datetime import datetime
from mmbench.common.registry import Registry

class Evaluator:
    def __init__(self,
                 image_length: int=0,
                 custom_cfg_path: str=None,
                 custom_functions: dict=dict(),
                 custom_dataset_functions: dict=dict(),
                 cfg_path: str=os.path.dirname(__file__)+'/config.yaml',
                 data_home_dir: str=None) -> None:
        """
        Args:
            custom_cfg_path (str, optional): _description_. Defaults to None.
            custom_functions (dict, optional): {"forward_step": func, ...} or {"task_name": {"forward_step": func, ...}, ...}
            custom_dataset_functions (dict, optional): {"normal_qa": func, ...} or {"task_name": {"normal_qa": func, ...}, ...}
            cfg_path (str, optional): _description_. Defaults to os.path.dirname(__file__)+'/config.yaml'.
        """
        self.default_cfg = OmegaConf.load(cfg_path)
        self._update_params(OmegaConf.load(custom_cfg_path))
        
        self.mmeval_home = os.environ.get("MMEVAL_HOME", os.path.join(os.path.expanduser('~'), ".mmbench_eval_tmp"))
        if not os.path.exists(self.mmeval_home):
            os.mkdir(self.mmeval_home, exist_ok=True)
            print_rank0(f"Using mmeval home: {self.mmeval_home}")
        self.data_home_dir = data_home_dir
        
        self.tasks = {
            name: Registry.get_task_class(name)(self.default_cfg["tasks"][level][name], \
                                                **{"custom_functions": self.get_custom_functions(custom_functions, name), \
                                                   "custom_dataset_functions": self.get_custom_functions(custom_dataset_functions, name), \
                                                   "image_length": image_length})
              for level in self.default_cfg["tasks"].keys() for name in self.default_cfg["tasks"][level]
        }
    
    def get_task_names(self):
        return Registry.list_tasks()
    
    def get_metric_names(self):
        return Registry.list_metrics()

    def get_custom_functions(self, custom_function_dict, task_name):
        ret = custom_function_dict[task_name] if task_name in custom_function_dict else {}
        task_names = set(self.get_task_names())
        for k,v in custom_function_dict.items():
            if k not in task_names:
                ret[k] = v
        return ret
        
    def _update_params(self, custom_params):
        """update self.default_cfg using custom_params
        """
        if 'tasks' in custom_params:
            for level_name in custom_params['tasks'].keys():
                for task_name, params in custom_params['tasks'][level_name].items():
                    self.default_cfg["tasks"][level_name][task_name].update(params)
        if 'params' in custom_params:
            self.default_cfg["params"].update(custom_params['params'])

    def evaluate(self, args, mt, eval_tasks=[]):
        all_scores = {}
        for k,v in self.default_cfg["params"].items():
            setattr(args, k, v)
        timestring = datetime.now().strftime("%m-%d-%H-%M")
        args.data_home_dir = self.data_home_dir
        args.save = args.save if hasattr(args, 'save') and args.save else self.mmeval_home
        args.save_result_path = os.path.join(args.save, f'{args.experiment_name}-{timestring}.jsonl')
        if not os.path.exists(args.save):
            print_rank0(f"Create save home: {args.save}")
            os.makedirs(args.save, exist_ok=True)
        # start
        failed_tasks = []
        for i, task_name in enumerate(eval_tasks):
            try:
                c_task = self.tasks[task_name]
                # reset args & model states
                args_cp = copy.deepcopy(args)
                print_rank0(f'Task ({i+1}/{len(eval_tasks)}) begin: {task_name}')
                # refine ckpt path
                args_cp.save_details_result_path = os.path.join(args.save,
                                                                f'{args.experiment_name}_{c_task.task_name}-{timestring}.csv')
                # evaluate
                all_scores[task_name] = c_task.do_evaluate(args_cp, mt)
                # save
                if torch.distributed.get_rank(group=mpu.get_data_parallel_group()) == 0:
                    with jsonlines.open(args.save_result_path, mode='a') as fp:
                        _tmp = {"task": task_name, "results": all_scores[task_name]}
                        fp.write(_tmp)
                print_rank0('-'*80)
                print_rank0(f'{task_name} results: {all_scores[task_name]}')
                print_rank0('-'*80)
                print_rank0(f'Task ({i+1}/{len(eval_tasks)}) end: {task_name}')
            except Exception as e:
                import traceback
                print_rank0(e, level=logging.ERROR)
                print_rank0(traceback.format_exc(), level=logging.ERROR)
                failed_tasks.append(task_name)
        print_rank0('Complete.')
        if len(failed_tasks) > 0:
            print_rank0(f'Failed Tasks: {failed_tasks}', level=logging.ERROR)
        if os.path.exists(args.save_result_path):
            print_rank0(f'Results are saved in {args.save_result_path}')
        return all_scores
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_tasks', type=str, nargs='+', help='Specify the tasks for evaluation')
    args = parser.parse_args()

    evaluator = Evaluator()
    print(evaluator.get_task_names())
    print(evaluator.get_metric_names())
    print("done")