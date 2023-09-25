import os
import csv
from typing import Dict
import collections

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask


@Registry.register_task('TDIUC')
class TDIUCTask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'TDIUC'
        
        # with open(os.path.join(os.path.dirname(__file__), 'sample_answerkey.csv')) as f:
        #    answerkey = csv.reader(f)
        #    self.answerkey = dict((rows[0],rows[1]) for rows in answerkey)

        super().__init__(task_cfg, **kw_args)
    
    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Avg"] = metric_cls.calc_scores(results_df["answer"], results_df["preds"])
        for c_type in results_df["question_type"].unique().tolist():
            c_df = results_df[results_df["question_type"] == c_type].drop_duplicates(subset=["question_id"])
            metrics_scores[c_type] = metric_cls.calc_scores(c_df["answer"], c_df["preds"])
        return metrics_scores
    
    # def calc_scores(self, args, results_df) -> Dict:

    #     metrics_scores = {}
        
    #     res_examples = {ex.idx: ex for ex in res_examples}
    #     result = collections.defaultdict(list)
    #     notfound_gt, notfound_res = 0, 0

    #     for name, record in results_df.iterrows():


    #         gt_answer = record['answer']
    #         gt_type = record['question_type']
    #         quesid = record['question_id']

    #         if gt_ex.idx in res_examples:
    #             res_ex = res_examples[gt_ex.idx]
    #             if gt_answer in self.answerkey:
    #                 gt_ans_idx = int(self.answerkey[gt_answer])
    #             else:
    #                 notfound_gt = 1
    #                 result[gt_type + '_f'].append(gt_ex.idx)
    #             if res_ex.answers[0] in self.answerkey:
    #                 pred_ans_idx = int(self.answerkey[res_ex.answers[0]])
    #             else:
    #                 notfound_res = 1
    #                 result[gt_type + '_f'].append(gt_ex.idx)

    #             if pred_ans_idx == gt_ans_idx:
    #                 result[gt_type + '_t'].append(gt_ex.idx)
    #             else:
    #                 result[gt_type + '_f'].append(gt_ex.idx)
    #         else:
    #             pred_ans_idx[gt_type + '_f'].append(gt_ex.idx)
    #     print(f"[TDIUC] {notfound_res}, {notfound_gt} examples from predictions and ground-truth are not found in answerkey, respectively.")

    #     types = list(set([ex.example_type for ex in self.examples]))
    #     sum_acc = []
    #     eps = 1e-10
    #     for tp in types:
    #         acc = 100*(len(result[tp'_t']) / len(result[tp'_t']  result[tp'_f']))
    #         sum_acc.append(acc  eps)
    #         metrics_scores["Acc for type "  tp] = acc
    #     metrics_scores["Acc sum"] = sum_acc

    #     return metrics_scores
