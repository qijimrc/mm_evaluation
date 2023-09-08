from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.tasks.base_task import BaseTask
from typing import Any, Dict, List
import os
import json



@Registry.register_task('VQAv2')
class VQAv2Task(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'VQAv2'
        super().__init__(task_cfg, custom_functions, **kw_args)

    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Acc"] = metric_cls.calc_scores(results_df["answer"], results_df["preds"])
        for c_type in results_df["question_type"].unique().tolist():
            c_df = results_df[results_df["question_type"] == c_type].drop_duplicates(subset=["question_id"])
            metrics_scores[c_type] = metric_cls.calc_scores(c_df["answer"], c_df["preds"])
        return metrics_scores
        


