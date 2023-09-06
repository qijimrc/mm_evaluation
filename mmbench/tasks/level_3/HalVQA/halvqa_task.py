import json
import random
import pandas as pd
from io import BytesIO
from PIL import Image
from typing import Dict, List

from sat.helpers import print_rank0
from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('HalVQA')
class HalVQATask(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'HalVQA'
        super().__init__(task_cfg, custom_functions, **kw_args)
    
    def process_fn_webDataset(self, args, mt, src):
        for data in src:
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
                prompt = qa["prompt"]
                text_dict = mt.text_processor(qa["answer"], prompt)
                if text_dict is None:
                    continue
                ret.update(text_dict)
                ret.update(img_dict)
                yield ret
    
    def calc_scores(self, args, result_df) -> Dict:
        # compute scores
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Total"] = metric_cls.calc_scores(result_df["answer"], result_df["preds"])
        for ext in result_df["eval_type"].unique().tolist():
            c_df = result_df[result_df["eval_type"] == ext].drop_duplicates(subset=["question_id"])
            metrics_scores[ext] = metric_cls.calc_scores(c_df["answer"], c_df["preds"])
        return metrics_scores
        