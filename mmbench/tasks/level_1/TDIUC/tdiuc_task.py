import os
import csv
import json
import random
import collections
import pandas as pd
from io import BytesIO
from PIL import Image
from collections import Counter
from typing import Dict, List
from torch.utils.data import Dataset
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask


@Registry.register_task('TDIUC')
class TDIUCTask(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'TDIUC'
        
        with open(os.path.join(os.path.dirname(__file__), 'sample_answerkey.csv')) as f:
           answerkey = csv.reader(f)
           self.answerkey = dict((rows[0],rows[1]) for rows in answerkey)

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
                    "label_text": qa["answer"]
                }
                # text
                text_dict = mt.text_processor(qa["answer"], qa["question"])
                if text_dict is None:
                    continue
                ret.update(text_dict)
                ret.update(img_dict)
                yield ret

    def calc_scores(self, args, results_total) -> Dict:
        mirror_df = self.get_data_mirror(args)

        etypes = set(mirror_df["question_type"])
        question_ids, preds, labels = results_total["question_ids"], results_total["preds"], results_total["labels"]
        res_df = pd.DataFrame({"question_ids": question_ids, "preds": preds, "labels": labels})
        
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Avg"] = metric_cls.calc_scores(res_df["labels"], res_df["preds"])
        for c_type in etypes:
            c_df = mirror_df[mirror_df["question_type"] == c_type].drop_duplicates(subset=["question_id"])
            c_df = res_df[res_df["question_ids"].isin(c_df["question_id"])]
            metrics_scores[c_type] = metric_cls.calc_scores(c_df["labels"], c_df["preds"])
        return metrics_scores
