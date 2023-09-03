from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.tasks.base_task import BaseTask
from typing import Any, Dict, List
import collections
import csv
import os
import json



# @Registry.register_task('TDIUC')
class TDIUCTask(BaseTask):
    def __init__(self, task_cfg):

        self.task_name = 'TDIUC'
        self.img_dir = task_cfg.img_dir
        self.anns_paths = task_cfg.anns_paths
        self.metrics = task_cfg.metrics
        
        with open(os.path.join(os.path.dirname(__file__), 'sample_answerkey.csv')) as f:
           answerkey = csv.reader(f)
           self.answerkey = dict((rows[0],rows[1]) for rows in answerkey)

        super().__init__(self.img_dir, self.anns_paths)

    def to_examples(self, img_dir: str, anns_paths: List) ->List[Example]:
        """ Convert annotations to canonical examples.
          Args:
            @img_dir: the root dir of vision source.
            @anns_paths: the paths of annotation files.
          Return:
            A list of examples instanced from the `Example` class.
        """
        examples = []
        with open(anns_paths) as f:
           for qa_info in json.load(f)['annotations']:
              ex = Example(task=self.task_name,
                          idx=qa_info['question_id'],
                          img_path=os.path.join(img_dir, 'COCO_val2014_{}{}.jpg'.format(''.join(['0']*(12-len(str(qa_info['image_id'])))), qa_info['image_id'])),
                          question=qa_info['question'],
                          answers=[ans['answer'] for ans in qa_info['answers']], # here ignored other answer information
                          example_type=qa_info['question_type'],
                          context='image_id=%s' % qa_info['image_id'] # used during evaluation
              )
              examples.append(ex)
        return examples

    def calc_scores(self, res_examples: List[Example], metrics: List[str]=['vqa_acc']) -> Dict:
        """ Calculate scores with specified metrics.
          Args:
            @examples:
            @metrics:
          Return:
            A result dict keyed by metrics names.
        """
        metrics_scores = {}
        
        res_examples = {ex.idx: ex for ex in res_examples}
        result = collections.defaultdict(list)
        notfound_gt, notfound_res = 0, 0
        for gt_ex in self.examples:
            gt_answer = gt_ex.answers[0]
            gt_type = gt_ex.example_type
            if gt_ex.idx in res_examples:
                res_ex = res_examples[gt_ex.idx]
                if gt_answer in self.answerkey:
                    gt_ans_idx = int(self.answerkey[gt_answer])
                else:
                    notfound_gt += 1
                    result[gt_type + '_f'].append(gt_ex.idx)
                if res_ex.answers[0] in self.answerkey:
                    pred_ans_idx = int(self.answerkey[res_ex.answers[0]])
                else:
                    notfound_res += 1
                    result[gt_type + '_f'].append(gt_ex.idx)

                if pred_ans_idx == gt_ans_idx:
                    result[gt_type + '_t'].append(gt_ex.idx)
                else:
                    result[gt_type + '_f'].append(gt_ex.idx)
            else:
                pred_ans_idx[gt_type + '_f'].append(gt_ex.idx)
        print(f"[TDIUC] {notfound_res}, {notfound_gt} examples from predictions and ground-truth are not found in answerkey, respectively.")

        types = list(set([ex.example_type for ex in self.examples]))
        sum_acc = []
        eps = 1e-10
        for tp in types:
            acc = 100*(len(result[tp+'_t']) / len(result[tp+'_t'] + result[tp+'_f']))
            sum_acc.append(acc + eps)
            metrics_scores["Acc for type " + tp] = acc
        metrics_scores["Acc sum"] = sum_acc

        return metrics_scores



