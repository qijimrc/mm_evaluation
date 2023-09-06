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
        self.etypes = ["existence", "color", "position"]
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
                    "label_text": qa["answer"]
                }
                prompt = qa["prompt"]
                text_dict = mt.text_processor(qa["answer"], prompt)
                if text_dict is None:
                    continue
                ret.update(text_dict)
                ret.update(img_dict)
                yield ret
    
    def calc_scores(self, args, results_total, metrics: List[str]=['acc']) -> Dict:
        mirror_df = self.get_data_mirror(args)

        metrics_scores = {}
        question_ids, preds, labels = results_total["question_ids"], results_total["preds"], results_total["labels"]
        res_df = pd.DataFrame({"question_ids": question_ids, "preds": preds, "labels": labels})
        # remove duplicates
        res_df = res_df.drop_duplicates(subset=["question_ids"])
        # compute scores
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Total"] = metric_cls.calc_scores(res_df["labels"], res_df["preds"])
        for ext in self.etypes:
            c_df = mirror_df[mirror_df["eval_type"] == ext].drop_duplicates(subset=["question_id"])
            c_df = res_df[res_df["question_ids"].isin(c_df["question_id"])]
            metrics_scores[ext] = metric_cls.calc_scores(c_df["labels"], c_df["preds"])
        return metrics_scores
        