from typing import Dict
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask
            
@Registry.register_task('GQA')
class GQATask(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'GQATask'
        super().__init__(task_cfg, custom_functions, **kw_args)

    def calc_scores(self, args, result_df) -> Dict:
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Total"] = metric_cls.calc_scores(result_df["answer"], result_df["preds"])
        return metrics_scores
        