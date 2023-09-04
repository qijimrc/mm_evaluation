import os
import pandas as pd
from PIL import Image
from typing import Dict, List
from torch.utils.data import Dataset
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

class HalVqaDataset(Dataset):
    def __init__(self, path, args, mt, mode, **kwargs):
        super().__init__()
        self.mt = mt
        self.mode = mode
        self.data = pd.read_csv(path)
        self.data_home_dir = args.data_home_dir

    def __getitem__(self, index):
        c_data = self.data.iloc[index]
        ret = {
            "question_id": str(c_data["question_id"]),
            "label_text": str(c_data["answer"])
        }
        # img
        image_path = os.path.join(self.data_home_dir, c_data["image"])
        img = Image.open(image_path).convert('RGB')
        img_dict = {'vision': self.mt.image_processor(img)}
        if self.mt.cross_image_processor:
            img_dict.update({'cross': self.mt.cross_image_processor(img)})
        ret.update(img_dict)
        # text
        prompt = c_data["prompt"]
        prompt = self.mt.text_processor.history_to_prompt([], prompt, add_eoi_first=True)
        text_dict = self.mt.text_processor(c_data["answer"], prompt)
        ret.update(text_dict)
        return ret
    
    def __len__(self):
        return len(self.data)

@Registry.register_task('HalVQA')
class HalVQATask(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'HalVQA'
        self.etypes = ["existence", "color", "position"]
        super().__init__(task_cfg, custom_functions, **kw_args)

    def create_dataset_function(self, mt, path, args):
        dataset = HalVqaDataset(path, args, mt, mode=self.mode)
        return dataset

    def calc_scores(self, args, results_total, metrics: List[str]=['acc']) -> Dict:
        if self.mode == "finetune":
            data_df = pd.read_csv(args.valid_data[0], dtype=str)
        elif self.mode == "test":
            data_df = pd.read_csv(args.test_data[0], dtype=str)
        metrics_scores = {}
        question_ids, preds, labels = results_total["question_ids"], results_total["preds"], results_total["labels"]
        res_df = pd.DataFrame({"question_ids": question_ids, "preds": preds, "labels": labels})
        # remove duplicates
        res_df = res_df.drop_duplicates(subset=["question_ids"])
        # compute scores
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Total"] = metric_cls.calc_scores(res_df["labels"], res_df["preds"])
        for ext in self.etypes:
            c_df = data_df[data_df["eval_type"] == ext].drop_duplicates(subset=["question_id"])
            c_df = res_df[res_df["question_ids"].isin(c_df["question_id"])]
            metrics_scores[ext] = metric_cls.calc_scores(c_df["labels"], c_df["preds"])
        return metrics_scores
        