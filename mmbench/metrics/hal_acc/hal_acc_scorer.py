from typing import Any
from mmbench.common.example import Example
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
from typing import List, Dict



@Registry.register_metric('hal_acc')
class HalAccMetric(BaseMetric):
    def __init__(self) -> None:
        self.uncertainty_words = ["不知道",
                                  "不确定",
                                  "抱歉",
                                  "无法得知",
                                  "无法确认",
                                  "如果能提供更多信息"]

    def _contains(self, pred:str, answer:str) -> bool:
        if answer in pred:
            return 1
        for word in self.uncertainty_words:
            if word in answer:
                return 0.4
        return 0

    @classmethod
    def calc_scores(cls, res_examples: List[Example], ans_examples: List[Example], eval_type=None) -> Dict:
        scores = {}
        predict_result = {ex.idx: ex.answers[0] for ex in res_examples}
        true_result = {ex.idx: ex.answers[0] for ex in ans_examples}
        compare_results = []
        for qid, pred in predict_result.items():
            assert qid in true_result
            true = true_result[qid]
            score = cls()._contains(pred, true)
            compare_results.append(score)
        scores['acc'] = sum(compare_results) / len(compare_results)
        return scores
