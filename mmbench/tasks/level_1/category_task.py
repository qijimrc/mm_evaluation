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
            
            if len(real_categories) >=2 and len(pred_categories) >=2:
                print(f"{answer}, {pred}")
                continue
                
            real_class += real_categories
            pred_class += pred_categories

        mlb = MultiLabelBinarizer()
        list_a_encoded = mlb.fit_transform(real_class)
        list_b_encoded = mlb.transform(pred_class)

        precision, recall, f1, _ = precision_recall_fscore_support(list_a_encoded, list_b_encoded, average=None)
        for label, p, r, f1 in zip(mlb.classes_, precision, recall, f1):
            metrics_scores[f"{label}_precision"] = round(p, 3)
            metrics_scores[f"{label}_recall"] = round(r, 3)
            metrics_scores[f"{label}_f1_score"] = round(f1, 3)            
    
        return metrics_scores


# def calc_scores(results_df, categories_dict) -> Dict:
#         metrics_scores = {}
#         real_class = []
#         pred_class = []
#         error_cnt = 0
#         error_cnt_total = 0
#         for i, r in results_df.iterrows():
#             answer, pred = r["answer"], r["preds"]
#             real_categories = answer.split("，")
#             pred_categories = pred.split("，")
            
#             index = 0
#             for categories in pred_categories:
#                 if categories not in categories_dict:
#                     error_cnt_total += 1
#                     index = 1
#             if index == 1:
#                 error_cnt += 1
            
#             if "扩展能力-梗图理解-梗图理解" in real_categories or "扩展能力-梗图理解-梗图理解" in pred_categories:
#                 print(r['question_id'], r['question'], r['answer'], r["preds"])
#             real_class.append(real_categories)
#             pred_class.append(pred_categories)
        
#         print(f"error cnt {error_cnt}, error cnt total label {error_cnt_total}")
#         mlb = MultiLabelBinarizer()
#         list_a_encoded = mlb.fit_transform(real_class)
#         list_b_encoded = mlb.transform(pred_class)

#         precision, recall, f1, _ = precision_recall_fscore_support(list_a_encoded, list_b_encoded, average=None)
#         for label, p, r, f1 in zip(mlb.classes_, precision, recall, f1):
#             metrics_scores[f"{label}_precision"] = round(p, 3)
#             metrics_scores[f"{label}_recall"] = round(r, 3)
#             metrics_scores[f"{label}_f1_score"] = round(f1, 3)            
    
#         return metrics_scores


# def get_category(filename):
#     data = json.load(open(filename, 'r'))
#     results = {}
#     for key, value in data['results'].items():
#         simplified_key = key.rsplit('_', 1)[0]
#         if simplified_key != "":
#             results[simplified_key] = True

#     return results


# if __name__ == "__main__":

#     import pandas as pd    
#     filename = "/home/wangyan/.mmbench_eval_tmp/evaluate-chatglm2_eva2sat_expert_and_cross_Category-11-15-03-37.csv"
#     category_name = "/home/wangyan/.mmbench_eval_tmp/evaluate-chatglm2_eva2sat_expert_and_cross-11-15-03-37.jsonl"
#     df = pd.read_csv(filename)
#     categories_dict = get_category(category_name)
#     metrics_scores = calc_scores(df, categories_dict)
#     print(metrics_scores)


                    