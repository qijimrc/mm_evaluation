from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.tasks.base_task import BaseTask
from typing import Dict, List
import os
import pandas as pd

@Registry.register_task('HalVQA')
class HalVQATask(BaseTask):
    def __init__(self, task_cfg):

        self.task_name = 'HalVQA'
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
        df = pd.read_csv(anns_paths, encoding='utf-8')
        for idx,row in df.iterrows():
            ex = Example(task=self.task_name,
                         idx=idx,
                         img_path=os.path.join(img_dir, row["image_path"]),
                         question=row["prompt"],
                         answers=[row["answer"]])
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
        