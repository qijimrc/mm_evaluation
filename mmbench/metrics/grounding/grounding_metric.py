from typing import Any
from typing import List, Dict
from mmbench.common.example import Example
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
from mmbench.metrics.grounding.utils import box_iou
import torch



@Registry.register_metric('grounding')
class GroundingMetric(BaseMetric):
    def __init__(self) -> None:
        pass


    @classmethod
    def calc_scores(self, res_examples: List[Example], gt_examples: List[Example]) -> Dict:
        """ Use official VQA evaluation script to report metrics.
          Args:
            @res_examples: a list of result examples instanced by `Example` class.
            @gt_examples: a list of ground truth examples.
          Return:
            the calculated metric scores.
        """
        scores = {}

        res_examples = {ex.idx:ex for ex in res_examples}
        gt_examples = {ex.idx:ex for ex in gt_examples}

        sum_accu, sum_iou, cnt_test = 0.0, 0.0, 0.0
        for idx, ex in gt_examples.items():
          gt_boxes = ex.answers
          res_boxes = res_examples[idx].answers
          if type(res_boxes) is not list:
              raise ValueError("The answers of each resulting example must be a list of boxes.")
          iou, union = box_iou(gt_boxes, res_boxes)
          iou = torch.diag(iou)
          # print(t, res['boxes'], iou, union)
          sum_accu = sum_accu + torch.sum((iou > 0.5).type(torch.float)).item()
          sum_iou = sum_iou + torch.sum(iou).item()
          cnt_test = cnt_test + len(gt_boxes)

        scores["accuracy_iou0.5"] = sum_accu / cnt_test
        scores["miou"] = sum_iou / cnt_test
        
        return scores