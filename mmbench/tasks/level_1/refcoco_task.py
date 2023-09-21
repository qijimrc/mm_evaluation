import os
import re
import json
import random
import pandas as pd
from PIL import Image
from io import BytesIO
from typing import Dict, List
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask


@Registry.register_task('RefCOCO')
class RefCoCoTask(BaseTask):
    def __init__(self, task_cfg,  **kw_args):
        self.task_name = 'RefCOCO'
        super().__init__(task_cfg,  **kw_args)

    def text_to_box(self, box_text):
        pattern = r"\[\[(.*?)\]\]"
        positions = re.findall(pattern, box_text)
        boxes = [[[int(y) for y in x.split(',')] for x in pos.split(';') if x.replace(',', '').isdigit()] for pos in positions]
        if len(boxes) == 0:
            return [[0.0, 0.0, 0.0, 0.0]]
        return boxes[0]

    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        # compute scores
        metric_cls = Registry.get_metric_class('grounding')
        results_df["answer_boxes"] = results_df["labels"].apply(lambda x: self.text_to_box(x))
        results_df["preds"] = results_df["preds"].apply(lambda x: self.text_to_box(x))
        for c_type in results_df["type"].unique().tolist():
            c_df = results_df[results_df["type"] == c_type].drop_duplicates(subset=["question_id"])
            metrics_scores[c_type] = metric_cls.calc_scores(c_df["answer_boxes"], c_df["preds"])
        return metrics_scores
        