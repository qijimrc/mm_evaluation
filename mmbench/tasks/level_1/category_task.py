from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.tasks.base_task import BaseTask
from typing import Any, Dict, List
import os
import json
from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

@Registry.register_task('Category')
class Category2Task(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'Category'
        super().__init__(task_cfg, **kw_args)

    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        real_class = []
        pred_class = []
        for i, r in results_df.iterrows():
            answer, pred = r["answer"], r["preds"]
            real_categories = answer.split("，")
            pred_categories = pred.split("，")
            if "，" not in pred and ',' in pred:
                pred_categories = pred.split(",")
            
            if len(real_categories) <3 and len(pred_categories) <3:
                print(f"{answer}, {pred}")
                continue
                
            real_class.append(real_categories)
            pred_class.append(pred_categories)

            mlb = MultiLabelBinarizer()
            list_a_encoded = mlb.fit_transform(real_class)
            list_b_encoded = mlb.transform(pred_class)

            precision, recall, f1, _ = precision_recall_fscore_support(list_a_encoded, list_b_encoded, average=None)
            for label, p, r, f1 in zip(mlb.classes_, precision, recall, f1):
                metrics_scores[f"{label}_precision"] = round(p, 3)
                metrics_scores[f"{label}_recall"] = round(r, 3)
                metrics_scores[f"{label}_f1_score"] = round(f1, 3)            
    
        return metrics_scores
        