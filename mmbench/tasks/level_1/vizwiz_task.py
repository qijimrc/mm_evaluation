from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask
from typing import Any, Dict, List
import os
import json



@Registry.register_task('VizWizVQA')
class VizWizVQATask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'VizWizVQA'
        super().__init__(task_cfg, **kw_args)

    def calc_scores(self, args, results_df) -> Dict:
        metric_cls = Registry.get_metric_class('vqa_acc')
        pred_qas = [{"question_id": r['question_id'], "answer": r["preds"]} for i, r in results_df.iterrows()]
        gt_qas = [{"question_id": r['question_id'], "answers": r["answer_list"]} for i, r in results_df.iterrows()]
        return metric_cls.calc_scores(pred_qas, gt_qas)
        