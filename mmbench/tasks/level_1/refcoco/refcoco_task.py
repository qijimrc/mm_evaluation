from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.tasks.base_task import BaseTask
from typing import Any, Dict, List
import os
import json



@Registry.register_task('RefCOCO')
class RefCOCO(BaseTask):
    def __init__(self, task_cfg):

        self.task_name = 'RefCOCO'
        self.img_dir = task_cfg.img_dir
        self.anns_paths = task_cfg.anns_paths
        self.metrics = task_cfg.metrics
        self.n_samples = task_cfg.n_samples
        self.seed = task_cfg.seed
        
        super().__init__(self.img_dir, self.anns_paths)

    def to_examples(self, img_dir: str, anns_paths: str) ->List[Example]:
        """ Convert annotations to canonical examples.
          Args:
            @img_dir: the root dir of vision source.
            @anns_paths: the paths of annotation files.
          Return:
            A list of examples instanced from the `Example` class.
        """
        examples = []
        with open(anns_paths) as f:
            sid = 0
            for line in f:
                ann_info = json.loads(line)
                ex = Example(task=self.task_name,
                            idx=sid,
                            # img_path=os.path.join(img_dir, ann_info['img_path']),
                            img_path=ann_info['img_path'],
                            question=ann_info['question'],
                            answers=[ann_info['target']],
                            example_type='REC',
                            context="sentence=%s" % ann_info['sentence']
                )
                examples.append(ex)
                sid += 1
        return examples

    def calc_scores(self, res_examples: List[Example], metrics: List[str]=['grounding']) -> Dict:
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
          if name == 'grounding':
            scores = metric_cls.calc_scores(res_examples, self.examples)
          metrics_scores[name] = scores
        return scores
    




@Registry.register_task('RefCOCOp')
class RefCOCO(BaseTask):
    def __init__(self, task_cfg):

        self.task_name = 'RefCOCOp'
        self.img_dir = task_cfg.img_dir
        self.anns_paths = task_cfg.anns_paths
        self.metrics = task_cfg.metrics
        self.n_samples = task_cfg.n_samples
        self.seed = task_cfg.seed
        
        super().__init__(self.img_dir, self.anns_paths)

    def to_examples(self, img_dir: str, anns_paths: str) ->List[Example]:
        """ Convert annotations to canonical examples.
          Args:
            @img_dir: the root dir of vision source.
            @anns_paths: the paths of annotation files.
          Return:
            A list of examples instanced from the `Example` class.
        """
        examples = []
        with open(anns_paths) as f:
            sid = 0
            for line in f:
                ann_info = json.loads(line)
                ex = Example(task=self.task_name,
                            idx=sid,
                            # img_path=os.path.join(img_dir, ann_info['img_path']),
                            img_path=ann_info['img_path'],
                            question=ann_info['question'],
                            answers=[ann_info['target']],
                            example_type='REC',
                            context="sentence=%s" % ann_info['sentence']
                )
                examples.append(ex)
                sid += 1
        return examples

    def calc_scores(self, res_examples: List[Example], metrics: List[str]=['grounding']) -> Dict:
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
          if name == 'grounding':
            scores = metric_cls.calc_scores(res_examples, self.examples)
          metrics_scores[name] = scores
        return scores



@Registry.register_task('RefCOCOg')
class RefCOCO(BaseTask):
    def __init__(self, task_cfg):

        self.task_name = 'RefCOCOg'
        self.img_dir = task_cfg.img_dir
        self.anns_paths = task_cfg.anns_paths
        self.metrics = task_cfg.metrics
        self.n_samples = task_cfg.n_samples
        self.seed = task_cfg.seed
        
        super().__init__(self.img_dir, self.anns_paths)

    def to_examples(self, img_dir: str, anns_paths: str) ->List[Example]:
        """ Convert annotations to canonical examples.
          Args:
            @img_dir: the root dir of vision source.
            @anns_paths: the paths of annotation files.
          Return:
            A list of examples instanced from the `Example` class.
        """
        examples = []
        with open(anns_paths) as f:
            sid = 0
            for line in f:
                ann_info = json.loads(line)
                ex = Example(task=self.task_name,
                            idx=sid,
                            # img_path=os.path.join(img_dir, ann_info['img_path']),
                            img_path=ann_info['img_path'],
                            question=ann_info['question'],
                            answers=[ann_info['target']],
                            example_type='REC',
                            context="sentence=%s" % ann_info['sentence']
                )
                examples.append(ex)
                sid += 1
        return examples

    def calc_scores(self, res_examples: List[Example], metrics: List[str]=['grounding']) -> Dict:
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
          if name == 'grounding':
            scores = metric_cls.calc_scores(res_examples, self.examples)
          metrics_scores[name] = scores
        return scores
