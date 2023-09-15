from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.tasks.base_task import BaseTask
from typing import Any, Dict, List
import os
import json



@Registry.register_task('VizWizVQA')
class VizWizVQATask(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'VizWizVQA'
        super().__init__(task_cfg, custom_functions, **kw_args)

    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('vqa_acc')
        metrics_scores["vqa_acc"] = metric_cls.calc_scores(results_df)
        return metrics_scores
        