import os
import json
from typing import Dict

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('NoCaps')
class NoCaps(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'NoCaps'
        super().__init__(task_cfg, custom_functions, **kw_args)
        
    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('caption')
        metrics_scores['acc'] = metric_cls.calc_scores(results_df["answer"], results_df["preds"])
        return metrics_scores
        # metric_cls = Registry.get_metric_class('caption')
        # # group results by image_id
        # label_dict, pred_dict = {}, {}
        # for image_id, sub_df in list(results_df.groupby(by="image_id")):
        #     label_dict[image_id] = sub_df["answer"].unique().tolist()
        #     pred_dict[image_id] = sub_df["preds"].unique().tolist()
        # metrics_scores = metric_cls.calc_scores(pred_dict, label_dict)
        # return metrics_scores
