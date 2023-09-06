import os
import random
import math
import pandas as pd
import json

from io import BytesIO
from PIL import Image
from typing import Dict, List
from torch.utils.data import Dataset
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('Visual7W')
class Visual7W(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'Visual7W'
        super().__init__(task_cfg, custom_functions, **kw_args)
    
    def process_fn_webDataset(self, args, mt, src):
        for data in src:
            # img
            try:
                img = Image.open(BytesIO(data['jpg'])).convert('RGB')
            except Exception as e:
                print_rank0(e)
                continue
            img_dict = {'vision': mt.image_processor(img)}
            if mt.cross_image_processor:
                img_dict.update({'cross': mt.cross_image_processor(img)})
            
            dialogues = json.loads(data['json'].decode("utf-8"))
            if args.data_mode == "train":
                dialogues = [random.choice(dialogues)]
            for qa in dialogues:
                ret = {
                    "question_id": qa["question_id"],
                }
                # text
                text_dict = mt.text_processor(qa["answer"], qa["prompt"])
                if text_dict is None:
                    continue
                ret.update(text_dict)
                ret.update(img_dict)
                yield ret

    def calc_scores(self, args, results_df) -> Dict:
        # compute score
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Avg"] = metric_cls.calc_scores(results_df["answer"], results_df["preds"])
        for etype in results_df["type"].unique().tolist():
            c_df = results_df[results_df["type"] == etype].drop_duplicates(subset=["question_id"])
            metrics_scores[etype] = metric_cls.calc_scores(c_df["answer"], c_df["preds"])
        return metrics_scores
