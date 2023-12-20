import os
import json
from typing import Dict

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('Infer')
class InferTask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'Infer'
        super().__init__(task_cfg, **kw_args)
        
    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        return metrics_scores
