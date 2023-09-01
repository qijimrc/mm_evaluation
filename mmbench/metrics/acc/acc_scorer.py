from sklearn.metrics import accuracy_score

from mmbench.common.example import Example
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
from typing import List, Dict

@Registry.register_metric('acc')
class Acc(BaseMetric):
    def __init__(self) -> None:
        pass

    @classmethod
    def calc_scores(cls, res_examples: List[Example], ans_examples: List[Example]) -> Dict:
        scores = {}
        true_results = [ex.answers for ex in ans_examples]
        pred_results = [ex.answers for ex in res_examples]
        scores['acc'] = accuracy_score(true_results, pred_results)
        return scores
