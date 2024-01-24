from typing import Dict
import json

from sat.helpers import print_rank0
from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('POPE')
class POPETask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'POPE'
        super().__init__(task_cfg, **kw_args)
    
    def calc_scores(self, args, result_df) -> Dict:
        metric_cls = Registry.get_metric_class('pope_score')
        pred_qas = [{"question_id": r['question_id'], "answer": r["preds"]} for i, r in result_df.iterrows()]
        return metric_cls.calc_scores(pred_qas)
