from typing import Dict

from sat.helpers import print_rank0
from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('HalVQA')
class HalVQATask(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'HalVQA'
        super().__init__(task_cfg, custom_functions, **kw_args)
    
    def calc_scores(self, args, result_df) -> Dict:
        # compute scores
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Total"] = metric_cls.calc_scores(result_df["answer"], result_df["preds"])
        for ext in result_df["eval_type"].unique().tolist():
            c_df = result_df[result_df["eval_type"] == ext].drop_duplicates(subset=["question_id"])
            metrics_scores[ext] = metric_cls.calc_scores(c_df["answer"], c_df["preds"])
        return metrics_scores
        