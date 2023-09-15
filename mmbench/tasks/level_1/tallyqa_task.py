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


@Registry.register_task('TallyQA')
class TallyQATask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'TallyQA'
        super().__init__(task_cfg, **kw_args)

    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["acc"] = metric_cls.calc_scores(results_df["answer"], results_df["preds"])
        return metrics_scores
        