from typing import Dict

from mmdoctor.common.logger import log
from mmdoctor.common.registry import Registry
from mmdoctor.metrics.base_metric import BaseMetric
from mmdoctor.metrics.vqa_acc.vqa_eval import VQAEval

@Registry.register_metric('vqa_acc')
class VqaAccMetric(BaseMetric):
    def __init__(self) -> None:
        pass

    @classmethod
    def calc_scores(cls, results_data):
        """
        Args:
            results_data (list of dict): [
                {
                    "question_id": str (required),
                    "predict": str (required),
                    "answer": str (required),
                    "answer_list": str (optional, the priority is higher than answer)
                }, ...
            ]
        Returns:
            scores (dict) : {
                avg_metrics: float,
                ...
            }
        """
        pred_qas = [{"question_id": r['question_id'], "answer": r["predict"]} for r in results_data]
        gt_qas = []
        for r in results_data:
            c_qas = {"question_id": r['question_id'], "answers": r["answer_list"]}
            # optional
            for item in ["question_type", "answer_type"]:
                if item in r:
                    c_qas[item] = r[item]
            gt_qas.append(c_qas)
        return cls.compute(pred_qas, gt_qas)

    @classmethod
    def compute(self, pred_qas, gt_qas) -> Dict:
        """ Use official VQA evaluation script to report metrics.
        Args:
            pred_qas (list of dict): each contains required keys of `question_id` and `answer`.
            gt_qas (list of dict): each contains required keys of `question_id`,  'answers' and  optional keys of `question`, `question_type` and 'answer_type'.
        Return:
            the calculated metric scores.
        """
        scores = {}

        vqa_scorer = VQAEval(pred_qas, gt_qas, n=2)
        log("Start VQA evaluation.")
        vqa_scorer.evaluate()
        
        # log accuracies
        overall_acc = vqa_scorer.accuracy["overall"]
        scores["avg_metrics"] = overall_acc

        for ans_type in vqa_scorer.accuracy["perAnswerType"]:
            log(
                "%s : %.02f"
                % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
            )
            scores[f'Atype-{ans_type}'] = vqa_scorer.accuracy["perAnswerType"][ans_type]

        for ans_type in vqa_scorer.accuracy["perQuestionType"]:
            log(
                "%s : %.02f"
                % (ans_type, vqa_scorer.accuracy["perQuestionType"][ans_type])
            )
            scores[f'Qtype-{ans_type}'] = vqa_scorer.accuracy["perQuestionType"][ans_type]

        return scores
