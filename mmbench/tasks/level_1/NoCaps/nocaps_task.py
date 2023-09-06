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

@Registry.register_task('NoCaps')
class NoCaps(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'NoCaps'
        with open(os.path.join(os.path.dirname(__file__), 'templates.json'), "r") as f:
            templates = json.load(f)
            self.cap_template = templates['Caption']
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
                if not self.no_prompt:
                    prompt = random.choice(self.cap_template).replace('<image>', '')
                else:
                    prompt = ""
                # text
                text_dict = mt.text_processor(qa["answer"], prompt)
                if text_dict is None:
                    continue
                ret.update(text_dict)
                ret.update(img_dict)
                yield ret

    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('caption')
        # group results by image_id
        label_dict, pred_dict = {}, {}
        for image_id, sub_df in list(results_df.groupby(by="image_id")):
            label_dict[image_id] = sub_df["answer"].unique().tolist()
            pred_dict[image_id] = sub_df["preds"].unique().tolist()
        metrics_scores = metric_cls.calc_scores(pred_dict, label_dict)
        return metrics_scores
