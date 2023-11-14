from typing import Dict

from sat.helpers import print_rank0
from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('TouchStone')
class TouchStoneTask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'TouchStoneTask'
        super().__init__(task_cfg, **kw_args)
    
    def calc_scores(self, args, result_df) -> Dict:
        result_df.to_csv(args.save_details_result_path, index=None)
        # compute scores
        metric_cls = Registry.get_metric_class('llm_score')
        scores = metric_cls(args).calc_scores(result_df, "english")
        return scores
        