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
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('vqa_acc')
        metrics_scores["Avg"] = metric_cls.calc_scores(results_df["answer_list"], results_df["preds"])
        for c_type in results_df["question_type"].unique().tolist():
            c_df = results_df[results_df["question_type"] == c_type].drop_duplicates(subset=["question_id"])
            metrics_scores[c_type] = metric_cls.calc_scores(c_df["answer_list"], c_df["preds"])
        return metrics_scores
