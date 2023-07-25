from src.common.registry import Registry
from src.common.example import Example
from src.tasks.base_task import BaseTask
from typing import Any, Dict, List
import os
import json



@Registry.register_task('VQAv2')
class VQAv2Task(BaseTask):
    def __init__(self, task_cfg):

        self.task_name = 'VQAv2'
        self.img_dir = task_cfg.img_dir
        self.anns_paths = task_cfg.anns_paths
        self.metrics = task_cfg.metrics

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
        idx = 0
        for ftype, path in anns_paths.items():
            if ftype == 'question':
              val_questions = {q['question_id']: q for q in json.load(open(path))['questions']}
            elif ftype == 'annotation':
              val_annotations = {a['question_id']: a for a in json.load(open(path))['annotations']}

        for qid in val_questions:
          ex = Example(task=self.task_name,
                      idx=idx,
                      img_path=os.path.join(img_dir, 'COCO_val2014_000000{}.jpg'.format(val_questions[qid]['image_id'])),
                      question=val_questions[qid]['question'],
                      answers=[ans['answer'] for ans in val_annotations[qid]['answers']]) # here ignored other answer informatio
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
            scores = metric_cls.calc_scores(res_examples, self.anns_paths['question'], self.anns_paths['annotation'])
          metrics_scores[name] = scores
        return scores
        


