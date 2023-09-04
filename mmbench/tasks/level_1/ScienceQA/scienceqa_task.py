import os
import pandas as pd
from PIL import Image
from typing import Dict, List
from torch.utils.data import Dataset
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

class ScienceqaDataset(Dataset):
    def __init__(self, path, args, mt, mode, **kwargs):
        super().__init__()
        self.mt = mt
        self.mode = mode
        self.data = pd.read_csv(path)
        self.data_home_dir = args.data_home_dir
        self.img_pad = os.path.join(os.path.dirname(__file__), "no_img.png")

    def __getitem__(self, index):
        c_data = self.data.iloc[index]
        ttype = c_data["ttype"]
        ret = {
            "question_id": int(c_data["question_id"]),
            "label_text": str(c_data["answer"])
        }
        # img
        if ttype == "IMG":
            image_path = os.path.join(self.data_home_dir, c_data["image_path"])
            img = Image.open(image_path).convert('RGB')
        else:
            img = Image.open(self.img_pad).convert('RGB')
        img_dict = {'vision': self.mt.image_processor(img)}
        if self.mt.cross_image_processor:
            img_dict.update({'cross': self.mt.cross_image_processor(img)})
        ret.update(img_dict)
        # text
        prompt = c_data["prompt"]
        if ttype == "TXT":
            context = c_data["context"]
            prompt = f"Context: {context}\n" + prompt
        prompt = self.mt.text_processor.history_to_prompt([], prompt, add_eoi_first=True)
        text_dict = self.mt.text_processor(c_data["answer"], prompt)
        ret.update(text_dict)
        return ret
    
    def __len__(self):
        return len(self.data)

@Registry.register_task('ScienceQA')
class ScienceQA(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'ScienceQA'
        self.ttypes = ["NO", "IMG", "TXT"]
        self.etypes = ["LAN", "NAT", "SOC", "G1-6", "G7-12"]
        super().__init__(task_cfg, custom_functions, **kw_args)
    
    def create_dataset_function(self, mt, path, args):
        dataset = ScienceqaDataset(path, args, mt, mode=self.mode)
        return dataset

    def calc_scores(self, args, results_total, metrics: List[str]=['acc']) -> Dict:
        """ Calculate scores with specified metrics.
          Args:
            @examples:
            @metrics:
          Return:
            A result dict keyed by metrics names.
        """
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
        metrics_scores["Avg"] = metric_cls.calc_scores(res_df["labels"], res_df["preds"])
        for ttype in self.ttypes: 
            c_df = data_df[data_df["ttype"] == ttype].drop_duplicates(subset=["question_id"])
            c_df = res_df[res_df["question_ids"].isin(c_df["question_id"])]
            metrics_scores[ttype] = metric_cls.calc_scores(c_df["labels"], c_df["preds"])
        # etypes
        img_df = data_df[data_df["ttype"] == "IMG"].drop_duplicates(subset=["question_id"])
        for etype in self.etypes:
            c_qids = [row["question_id"] for i, row in img_df.iterrows() if etype in row["etype"]]
            c_df = res_df[res_df["question_ids"].isin(set(c_qids))]
            metrics_scores[etype] = metric_cls.calc_scores(c_df["labels"], c_df["preds"])
        return metrics_scores
