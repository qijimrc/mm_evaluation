from typing import Any
from mmbench.common.example import Example
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
from mmbench.metrics.vqa_acc.vqa_eval import VQAEval
from typing import List, Dict
from pathlib import Path
import re
import json



@Registry.register_metric('pope_score')
class POPEMetric(BaseMetric):
    def __init__(self) -> None:
        pass


    @classmethod
    def calc_scores(self, pred_qas) -> Dict:
    # def calc_scores(self, result_df) -> Dict:
        """ Use official POPE evaluation script to report metrics.
          Args:
            @pred_qas: a list of dict where each contains required keys of `question_id` and `answer`.
          Return:
            the calculated metric scores.
        """
        label_file = Path(__file__).absolute().parent/'coco_pope_popular.json'

        qid2answers = {item['question_id']: item['answer'] for item in pred_qas}
        label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]
        full_label = [json.loads(q) for q in open(label_file, 'r')]
        if len(pred_qas) != len(label_list):
            raise ValueError(f'The length of preds {len(pred_qas)} do not match the gt length {len(label_list)}')


        answers = []
        for item in full_label:
            idx = item['question_id']
            text = qid2answers[str(idx)]
            if text.find('.') != -1:
                text = text.split('.')[0]

            text = text.replace(',', '')
            words = text.split(' ')
            if 'No' in words or 'not' in words or 'no' in words:
                answers.append('no')
            else:
                answers.append('yes')


        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        pred_list = []
        for answer in answers:
            if answer == 'no':
                pred_list.append(0)
            else:
                pred_list.append(1)

        pos = 1
        neg = 0
        yes_ratio = pred_list.count(1) / len(pred_list)

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        print('TP\tFP\tTN\tFN\t')
        print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2*precision*recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        # print('Accuracy: {}'.format(acc))
        # print('Precision: {}'.format(precision))
        # print('Recall: {}'.format(recall))
        # print('F1 score: {}'.format(f1))
        # print('Yes ratio: {}'.format(yes_ratio))
        metrics = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1 score': f1, 'Yes ratio': yes_ratio}
        return metrics