import os
import json
import random
import pandas as pd
from io import BytesIO
from PIL import Image
from functools import partial
from typing import Dict, List
from sat.helpers import print_rank0
from sat.data_utils.webds import SimpleDistributedWebDataset

from mmbench.common.utils import get_tar_files
from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

def process_fn_GQAWebDataset(data_mode, args, mt,  src):
    for data in src:
        img_bytes = data['png'] if 'png' in data else data['jpg']
        try:
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(e)
            continue
        img_dict = {'vision': mt.image_processor(img)}
        if mt.cross_image_processor:
            img_dict.update({'cross': mt.cross_image_processor(img)})
        
        dialogues = json.loads(data['json'].decode("utf-8"))
        if data_mode == "train":
            dialogues = [random.choice(dialogues)]
        for qa in dialogues:
            ret = {
                "question_id": qa["question_id"],
                "label_text": qa["txt"]
            }
            prompt = "Question: {} Short answer:".format(qa["prompt"])
            text_dict = mt.text_processor(qa["txt"], prompt)
            if text_dict is None:
                continue
            ret.update(text_dict)
            ret.update(img_dict)
            yield ret
            
@Registry.register_task('GQA')
class GQATask(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'GQATask'
        super().__init__(task_cfg, custom_functions, **kw_args)

    def create_dataset_function(self, mt, path, args):
        path, data_mode = path.split("###")
        urls = get_tar_files(path)
        if hasattr(args, "random_urls") and args.random_urls:
            urls = random.shuffle(urls)
        dataset = SimpleDistributedWebDataset(urls, partial(process_fn_GQAWebDataset, data_mode, args, mt), args.seed)
        return dataset

    def calc_scores(self, args, results_total, metrics: List[str]=['acc']) -> Dict:
        metrics_scores = {}
        question_ids, preds, labels = results_total["question_ids"], results_total["preds"], results_total["labels"]
        res_df = pd.DataFrame({"question_ids": question_ids, "preds": preds, "labels": labels})
        # remove duplicates
        res_df = res_df.drop_duplicates(subset=["question_ids"])
        # compute scores
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Total"] = metric_cls.calc_scores(res_df["labels"], res_df["preds"])
        return metrics_scores
        