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

if __name__ == "__main__":
    metrics = VqaAccMetric()
    pred_qas = [{'question_id': '39450', 'answer': 'toronto'},
                {'question_id': '39451', 'answer': 'toronto'},
                {'question_id': '35065', 'answer': '10:10'},
                {'question_id': '36985', 'answer': 'working class'},
                {'question_id': '36986', 'answer': 'alan'}]
    gt_qas = [{'question_id': '39450', 'answers': ['toronto', 'toronto', 'toronto', 'toronto maple leafs', 'toronto', 'toronto ', 'toronto ', 'toronto', 'toronto maple leafs', 'toronto']},
              {'question_id': '39451', 'answers': ['toronto', 'toronto', 'toronto', 'toronto', 'toronto ', 'toronto', 'toronto', 'toronto', 'toronto', 'toronto']},
              {'question_id': '35065', 'answers': ['11:44', '11:45', 'answering does not r... the image', '11:44', '8:59', '8:58', '8:56', '11:43', '11:43', '11:44']},
              {'question_id': '36985', 'answers': ['old labour', 'working class', 'working class', 'new labour', 'working', 'working class', 'working class', 'unanswerable', 'working', 'lucky strike']},
              {'question_id': '36986', 'answers': ['alan', 'alan', 'alan', 'alan', 'alan', 'alan', 'alan', 'alan', 'alan', 'alan']}]
    import pandas as pd
    df = pd.read_csv("/zhipu-data/home/yuwenmeng/.mmbench_eval_tmp/evaluate-chatglm2_evahf_TextVQA-01-07-20-22.csv")
    pred_qas, gt_qas = [], []
    for i,r in df.iterrows():
        pred_qas.append({'question_id': r['question_id'], 'answer': r['preds']})
        gt_qas.append({'question_id': r['question_id'], 'answers': eval(r['answer_list'])})
    print(metrics.calc_scores(pred_qas, gt_qas))