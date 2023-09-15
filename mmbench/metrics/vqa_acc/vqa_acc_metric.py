from typing import Any
from mmbench.common.example import Example
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
from mmbench.metrics.vqa_acc.vqa_eval import VQAEval
from typing import List, Dict
import re



@Registry.register_metric('vqa_acc')
class VqaAccMetric(BaseMetric):
    def __init__(self) -> None:
        pass


    @classmethod
    def calc_scores(self, pred_qas, gt_qas) -> Dict:
    # def calc_scores(self, result_df) -> Dict:
        """ Use official VQA evaluation script to report metrics.
          Args:
            @pred_qas: a list of dict where each contains required keys of `question_id` and `answer`.
            @gt_qas: a list of dict where each contains required keys of `question_id`,  'answers' and  optional keys of `question`, `question_type` and 'answer_type'.
          Return:
            the calculated metric scores.
        """
        scores = {}

        vqa_scorer = VQAEval(pred_qas, gt_qas, n=2)
        print("Start VQA evaluation.")
        vqa_scorer.evaluate()

        # print accuracies
        overall_acc = vqa_scorer.accuracy["overall"]
        scores["avg_metrics"] = overall_acc

        for ans_type in vqa_scorer.accuracy["perAnswerType"]:
            print(
                "%s : %.02f"
                % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
            )
            scores[f'Atype-{ans_type}'] = vqa_scorer.accuracy["perAnswerType"][ans_type]

        for ans_type in vqa_scorer.accuracy["perQuestionType"]:
            print(
                "%s : %.02f"
                % (ans_type, vqa_scorer.accuracy["perQuestionType"][ans_type])
            )
            scores[f'Qtype-{ans_type}'] = vqa_scorer.accuracy["perQuestionType"][ans_type]

        return scores