from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.tasks.base_task import BaseTask
from typing import Any, Dict, List
import os
import json



# @Registry.register_task('VQAv2')
class VQAv2Task(BaseTask):
    def __init__(self, task_cfg):

        self.task_name = 'VQAv2'
        self.img_dir = task_cfg.img_dir
        self.anns_paths = task_cfg.anns_paths
        self.metrics = task_cfg.metrics
        
        with open(self.anns_paths) as f:
          ann = json.load(f)
          ann.pop('annotations')
          self.vqav2_info = ann

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
        for name in metrics:
          metric_cls = Registry.get_metric_class(name)
          if name == 'vqa_acc':
            scores = metric_cls.calc_scores(res_examples, self.examples, self.vqav2_info)
          metrics_scores[name] = scores
        return scores
        


