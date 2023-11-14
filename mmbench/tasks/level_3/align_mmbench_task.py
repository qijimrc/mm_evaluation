from typing import Dict

from sat.helpers import print_rank0
from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('AlignMMBench')
class AlignMMBenchTask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'AlignMMBench'
        super().__init__(task_cfg, **kw_args)
    
    def calc_scores(self, args, result_df) -> Dict:
        # compute scores
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        result_df["answer"] = result_df["answer"].apply(lambda x: chr(ord('A') + int(x)))
        metrics_scores["Total"] = metric_cls.calc_scores(result_df["answer"], result_df["preds"])
        for ext in result_df["type"].unique().tolist():
            c_df = result_df[result_df["type"] == ext].drop_duplicates(subset=["question_id"])
            metrics_scores[ext] = metric_cls.calc_scores(c_df["answer"], c_df["preds"])
        return metrics_scores
        