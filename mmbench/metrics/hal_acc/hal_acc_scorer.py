from typing import Any
from mmbench.common.example import Example
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
from typing import List, Dict



@Registry.register_metric('hal_acc')
class HalAccMetric(BaseMetric):
    def __init__(self) -> None:
        pass

    @classmethod
    def _contains(cls, pred:str, answer:str) -> bool:
        """if pred contained by answer, return True; or False.
        """
        return pred in answer

    @classmethod
    def calc_scores(cls, res_examples: List[Example], ans_examples: List[Example], eval_type=None) -> Dict:
        scores = {}
        predict_result = {ex.idx: ex.answers[0] for ex in res_examples}
        true_result = {ex.idx: ex.answers[0] for ex in ans_examples}
        compare_results = []
        for qid, pred in predict_result.items():
            assert qid in true_result
            true = true_result[qid]
            if cls._contains(pred, true):
                compare_results.append(1)
            else:
                compare_results.append(0)
        scores['acc'] = sum(compare_results) / len(compare_results)
        return scores
