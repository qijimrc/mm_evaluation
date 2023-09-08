from typing import Dict
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('Visual7W')
class Visual7W(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'Visual7W'
        super().__init__(task_cfg, custom_functions, **kw_args)
    
    def calc_scores(self, args, results_df) -> Dict:
        # compute score
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Avg"] = metric_cls.calc_scores(results_df["answer"], results_df["preds"])
        for etype in results_df["type"].unique().tolist():
            c_df = results_df[results_df["type"] == etype].drop_duplicates(subset=["question_id"])
            metrics_scores[etype] = metric_cls.calc_scores(c_df["answer"], c_df["preds"])
        return metrics_scores
