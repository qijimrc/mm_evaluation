import os
import sys
import copy
import torch
import logging
import jsonlines

if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from omegaconf import OmegaConf
from datetime import datetime

from mmdoctor.tasks.base_task import BaseTask
from mmdoctor.common.registry import Registry
from mmdoctor.common.logger import log

class Evaluator:
    def __init__(self, custom_data_hooks: dict=dict()):
        """
        Args:
            custom_data_hooks (dict, optional): {"task_name": {"normal_qa": func, ...}, ...}
        """
        self.default_cfg = OmegaConf.load(os.path.dirname(__file__)+'/config.yaml')
        self.mmeval_home = os.environ.get("MMEVAL_HOME", \
            os.path.join(os.path.expanduser('~'), ".mmbench_eval_tmp"))
        if not os.path.exists(self.mmeval_home):
            os.mkdir(self.mmeval_home)
            log(f"Using mmeval home: {self.mmeval_home}")
        # registry Task
        self.tasks = {}
        for task_type in self.default_cfg["tasks"].keys():
            for name in self.default_cfg["tasks"][task_type]:
                if not Registry.get_task_class(name):
                    Registry.register_task(name)(BaseTask)
                self.tasks[name] = Registry.get_task_class(name)(self.default_cfg["tasks"][task_type][name],
                                                                 custom_data_hooks=self.get_custom_functions(custom_data_hooks, name))

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
        
    def _update_params(self, custom_params):
        """update self.default_cfg using custom_params
        """
        if 'tasks' in custom_params:
            for level_name in custom_params['tasks'].keys():
                for task_name, params in custom_params['tasks'][level_name].items():
                    if task_name not in self.default_cfg["tasks"][level_name]:
                        self.default_cfg["tasks"][level_name][task_name] = {}
                    self.default_cfg["tasks"][level_name][task_name].update(params)

    def _evaluate_tasks(self, args, model_cls):
        model_scores = {}
        # start
        failed_tasks = []
        for i, task_name in enumerate(args.eval_tasks):
            try:
                log.set_task_name(task_name)
                c_task = self.tasks[task_name]
                log(f'Start ({i+1}/{len(args.eval_tasks)})')
                # reset args & model states
                args_cp = copy.deepcopy(args)
                args_cp.save_details_result_path = os.path.join(args_cp.save,
                                                                f'{log.get_model_name()}-{log.get_task_name()}')
                log(f"Detailed results will be saved in {args_cp.save_details_result_path}")
                # evaluate
                model_scores[log.get_task_name()] = c_task.do_evaluate(args_cp, model_cls)
                # save
                if args.rank == 0:
                    with jsonlines.open(args.save_result_path, mode='a') as fp:
                        _tmp = {"model": log.get_model_name(), "task": log.get_task_name(), "results": model_scores[task_name]}
                        fp.write(_tmp)
                log(f'Score: {model_scores[log.get_task_name()]}', add_sep_lines=True, sep_char='-')
                log(f'End ({i+1}/{len(args.eval_tasks)})')
            except Exception as e:
                import traceback
                log(traceback.format_exc(), level=logging.ERROR)
                failed_tasks.append(task_name)
        log('Complete.')
        if len(failed_tasks) > 0:
            log(f'Failed Tasks: {failed_tasks}', level=logging.WARNING)
        log.reset_model_name()
        return model_scores

    def run(self, args, model_names):
        # update args
        timestring = datetime.now().strftime("%Y-%m-%d-%H-%M")
        args.save = os.path.join(args.save or self.mmeval_home, f'mmevaluation-{timestring}')
        args.save_result_path = f"{args.save}/scores.jsonl"
        log(f"All detail results will be saved in {args.save} and evaluate scores will be saved in {args.save_result_path}...")
        if not os.path.exists(args.save):
            log(f"Create save dir: {args.save}")
            os.makedirs(args.save, exist_ok=True)
        # evaluate models
        all_scores = {}
        if not isinstance(model_names, list):
            model_names = [model_names]
        for i, model_cls in enumerate(model_names):
            if isinstance(model_cls, str):
                log.set_model_name(model_cls)
                model_cls = Registry.get_model_class(model_cls)(self.default_cfg['models'][model_cls], 
                                                                args)
            else:
                log.set_model_name(model_cls.__class__.__name__)
            if not hasattr(model_cls, "generate"):
                log("Not exist `generate` method!", level=logging.ERROR)
            else:
                # evaluate
                all_scores[log.get_model_name()] = self._evaluate_tasks(args, model_cls)
            log(f'Model ({i+1}/{len(model_names)}) end: {log.get_model_name()}')
        # TODO: results merge
        log(f'Results are saved in {args.save_result_path}')
        log('DONE.')
        return all_scores
