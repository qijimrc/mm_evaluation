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


@Registry.register_task('STVQA')
class STVQATask(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'STVQA'
        super().__init__(task_cfg, custom_functions, **kw_args)

    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('ANLS')
        pred_qas = {r["question_id"]: r["preds"] for i, r in results_df.iterrows()}
        gt_qas = {r["question_id"]: r["answer_list"] for i, r in results_df.iterrows()}
        metrics_scores["ANLS"] = metric_cls.calc_scores(pred_qas, gt_qas)
        return metrics_scores
        