from typing import Dict
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('ScienceQA')
class ScienceQA(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'ScienceQA'
        self.ttypes = ["NO", "IMG", "TXT"]
        self.etypes = ["LAN", "NAT", "SOC", "G1-6", "G7-12"]
        super().__init__(task_cfg, custom_functions, **kw_args)
    
    def calc_scores(self, args, results_df) -> Dict:
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('acc')
        metrics_scores["Avg"] = metric_cls.calc_scores(results_df["answer"], results_df["preds"])
        for ttype in self.ttypes: 
            c_df = results_df[results_df["ttype"] == ttype].drop_duplicates(subset=["question_id"])
            metrics_scores[ttype] = metric_cls.calc_scores(c_df["answer"], c_df["preds"])
        # etypes
        img_df = results_df[results_df["ttype"] == "IMG"].drop_duplicates(subset=["question_id"])
        for etype in self.etypes:
            c_qids = [row["question_id"] for i, row in img_df.iterrows() if etype in row["etype"]]
            c_df = results_df[results_df["question_ids"].isin(set(c_qids))]
            metrics_scores[etype] = metric_cls.calc_scores(c_df["answer"], c_df["preds"])
        return metrics_scores
