from typing import Dict

from mmdoctor.common.registry import Registry
from mmdoctor.tasks.base_task import BaseTask

@Registry.register_task('ScienceQA')
class ScienceQATask(BaseTask):
    def __init__(self, task_cfg,  **kw_args):
        self.task_name = 'ScienceQA'
        self.ttypes = ["NO", "IMG", "TXT"]
        self.etypes = ["LAN", "NAT", "SOC", "G1-6", "G7-12"]
        super().__init__(task_cfg, **kw_args)
    
    def calc_scores(self, args, results_data) -> Dict:
        metrics_scores = {}
        metric_cls = Registry.get_metric_class('em-acc')
        for c_data in results_data:
            c_data["answer"] = chr(ord('A') + int(c_data["answer"]))
        metrics_scores["Avg"] = metric_cls.calc_scores(results_data)
        for ttype in self.ttypes: 
            c_results_data = [v for v in results_data if v["ttype"] == ttype]
            metrics_scores[ttype] = metric_cls.calc_scores(c_results_data)
        # etypes
        results_data = [v for v in results_data if v["ttype"] == "IMG"]
        for etype in self.etypes:
            c_results_data = [v for v in results_data if etype in v["etype"]]
            metrics_scores[etype] = metric_cls.calc_scores(c_results_data)
        return metrics_scores
