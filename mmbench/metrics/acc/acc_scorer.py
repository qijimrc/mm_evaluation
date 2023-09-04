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
    def calc_scores(cls, trues, preds) -> Dict:
        assert len(trues) == len(preds)
        if len(trues) == 0:
            return 0.0
        acc = accuracy_score(trues, preds)
        return round(acc, 4)
