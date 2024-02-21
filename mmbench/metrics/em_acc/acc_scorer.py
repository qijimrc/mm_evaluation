from sklearn.metrics import accuracy_score

from mmdoctor.common.registry import Registry
from mmdoctor.metrics.base_metric import BaseMetric
from typing import List, Dict

@Registry.register_metric('em-acc')
class ExactMatchAcc(BaseMetric):
    def __init__(self) -> None:
        pass

    @classmethod
    def calc_scores(cls, results_data) -> Dict:
        """
        Args:
            results_data (list of dict): [
                {
                    "answer": str (required),
                    "predict": str (required),
                    "question_type": str (optional),
                    "answer_type": str (optional)
                }, ...
            ]
        Returns:
            report_acc (dict) : {
                "avg": float,
                ...
            }
        """
        def compute_score_per_type(type_name, flag_name):
            c_reports = {}
            result_types = {}
            for data in results_data:
                c_type = data[type_name]
                if c_type not in result_types:
                    result_types[c_type] = {"predict": [], "answer": []}
                result_types[c_type]["predict"].append(data["predict"])
                result_types[c_type]["answer"].append(data["answer"])
            for c_type, data in result_types.items():
                c_reports[f'{flag_name}-{c_type}'] = round(accuracy_score(data["answer"], data["predict"]) * 100, 2)
            return c_reports
            
        if len(results_data) == 0:
            return 0.0
        report_acc = {}
        preds = [v["predict"] for v in results_data]
        trues = [v["answer"] for v in results_data]
        report_acc["avg"] = round(accuracy_score(trues, preds) * 100, 2)
        # split types
        if 'question_type' in results_data[0]:
            report_acc.update(compute_score_per_type("question_type", "Qtype"))
        if 'answer_type' in results_data[0]:
            report_acc.update(compute_score_per_type("answer_type", "Atype"))
        return report_acc
