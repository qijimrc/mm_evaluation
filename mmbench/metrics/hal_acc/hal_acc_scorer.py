from sklearn.metrics import f1_score, accuracy_score

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
        self.uncertainty_score = 0.4

    def is_uncertainty_response(self, response) -> bool:
        for word in self.uncertainty_words:
            if word in response:
                return True
        return False

    def is_contains(self, pred:str, answer:str) -> bool:
        answer_list = answer.split(",")
        for answer in answer_list:
            if answer in pred:
                return True
        return False

    @classmethod
    def calc_scores(cls, res_examples: List[Example], ans_examples: List[Example], eval_type=None) -> Dict:
        scores, self = {}, cls()

        predict_result, true_result = [], []
        predict_result_dict = {ex.idx: ex.answers[0] for ex in res_examples}
        true_result_dict = {ex.idx: ex.answers[0] for ex in ans_examples}
        for key, value in predict_result_dict.items():
            assert key in true_result_dict
            predict_result.append(value)
            true_result.append(true_result_dict[key])
            
        true_class_map = {v: i for i, v in enumerate(list(set(true_result)))}
        true_classes = [true_class_map[v] for v in true_result]
        predict_classes, sample_weight = [], []
        for pred, true in zip(predict_result, true_result):
            if self.is_uncertainty_response(pred):
                sample_weight.append(self.uncertainty_score)
                predict_classes.append(true_class_map[true])
            else:
                sample_weight.append(1.0)
                if self.is_contains(pred, true):
                    predict_classes.append(true_class_map[true])
                else:
                    tmp = max(0, true_class_map[true] - 1) if true_class_map[true] > 0 else \
                        true_class_map[true] + 1
                    predict_classes.append(tmp)
        scores['acc'] = accuracy_score(true_classes, predict_classes, sample_weight=sample_weight)
        scores['f1'] = f1_score(true_classes, predict_classes, sample_weight=sample_weight, average='macro')
        return scores
