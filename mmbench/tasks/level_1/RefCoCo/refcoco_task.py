import os
import json
import random
import pandas as pd
from PIL import Image
from io import BytesIO
from typing import Dict, List
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask


@Registry.register_task('RefCoCo')
class RefCoCoTask(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'RefCoCo'
        super().__init__(task_cfg, custom_functions, **kw_args)

    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        # compute scores
        metric_cls = Registry.get_metric_class('grounding')
        results_df["answer_boxes"] = results_df["boxes"].apply(lambda x: x[0])
        for c_type in results_df["type"].unique().tolist():
            c_df = results_df[results_df["type"] == c_type].drop_duplicates(subset=["question_id"])
            metrics_scores[c_type] = metric_cls.calc_scores(c_df["answer_boxes"], c_df["preds"])
        return metrics_scores
        