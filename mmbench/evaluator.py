import os
import sys
import copy
import argparse

if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mmbench.common.registry import Registry
from mmbench.common.example import Example
from omegaconf import OmegaConf
from typing import Dict, List, Generator
from pprint import pprint


class Evaluator:
    def __init__(self, custom_cfg_path: str=None, custom_functions: dict=dict(), cfg_path: str=os.path.dirname(__file__)+'/config.yaml') -> None:
        """
        Args:
            custom_cfg_path (str, optional): _description_. Defaults to None.
            custom_functions (dict, optional): {"forward_step": func, ...} or {"task": {"forward_step": func, ...}, ...}
            cfg_path (str, optional): _description_. Defaults to os.path.dirname(__file__)+'/config.yaml'.
        """
        self.default_cfg = OmegaConf.load(cfg_path)
        self._update_params(OmegaConf.load(custom_cfg_path))

        self.finetuned_tmp_dir = os.environ.get("MMEVAL_HOME", os.path.join(os.path.expanduser('~'), ".mmbench_eval_tmp"))
        self.data_home_dir = self.default_cfg["home_env"][self.default_cfg["server_addr"]]["data_home"]
        
        self.tasks = {
            name: Registry.get_task_class(name)(self.default_cfg["tasks"][level][name], custom_functions[name] if name in custom_functions else custom_functions)
              for level in self.default_cfg["tasks"].keys() for name in self.default_cfg["tasks"][level]
        }
    
    def get_task_names(self):
        return Registry.list_tasks()
        
    def _update_params(self, custom_params):
        """update self.default_cfg using custom_params
        """
        if 'tasks' in custom_params:
            for level_name in custom_params['tasks'].keys():
                for task_name, params in custom_params['tasks'][level_name].items():
                    tmp_params = copy.deepcopy(params)
                    for spec_name in ["finetune_params", "eval_params"]:
                        if spec_name in tmp_params:
                            for pn, value in tmp_params[spec_name].items():
                                self.default_cfg['tasks'][level_name][task_name][spec_name][pn] = value
                        tmp_params.pop(spec_name)
                    for key, value in tmp_params.items():
                        self.default_cfg['tasks'][level_name][task_name][key] = value

    def evaluate(self, args, mt, eval_tasks=[]):
        all_scores = {}
        args.data_home_dir = self.data_home_dir
        for task_name in eval_tasks:
            c_task = self.tasks[task_name]
            if c_task.need_finetune:
                args.save = args.save if hasattr(args, 'save') and args.save else self.finetuned_tmp_dir
                c_task.do_finetune(args, mt)
            if c_task.need_evaluate:
                # TODO: load fintuned model
                all_scores[task_name] = c_task.do_evaluate(args, mt)
        return all_scores
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_tasks', type=str, nargs='+', help='Specify the tasks for evaluation')
    args = parser.parse_args()
    evaluator = Evaluator()