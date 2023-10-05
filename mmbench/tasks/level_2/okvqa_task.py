from typing import Dict
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask


@Registry.register_task('OKVQA')
class OKVQATask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'OKVQA'
        super().__init__(task_cfg, **kw_args)
    
    def calc_scores(self, args, results_df) -> Dict:
        metric_cls = Registry.get_metric_class('vqa_acc')
        pred_qas = [{"question_id": r['question_id'], "answer": r["preds"]} for i, r in results_df.iterrows()]
        gt_qas = [{"question_id": r['question_id'], "answers": eval(r["answer_list"]), "question_type": r["question_type"]} for i, r in results_df.iterrows()]
        return metric_cls.calc_scores(pred_qas, gt_qas)
