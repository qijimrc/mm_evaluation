from typing import Any, Dict, List
import os
import json

from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('OCRVQA')
class OCRVQA(BaseTask):
    def __init__(self, task_cfg):

        self.task_name = 'OCRVQA'
        self.img_dir = task_cfg.img_dir
        self.anns_paths = task_cfg.anns_paths
        self.metrics = task_cfg.metrics
        self.all_eval_types = set()
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
        
        drop_num = 0
        with open(anns_paths) as fp:
            sid = 0
            data = json.load(fp)
            for key, value in data.items():
              if value["split_str"] != "test":
                continue
              image_path = None
              if value["image"]:
                  image_path = os.path.join(self.img_dir, value["image"])
                  if not os.path.exists(image_path):
                    print(f"image not found: {image_path}, will be skipped.")
                    drop_num += 1
                    continue
              questions, answers = value["questions"], value["answers"]
              self.all_eval_types.add(value["genre"])
              for question, answer in zip(questions, answers):
                  ex = Example(task=self.task_name,
                              idx=sid,
                              img_path=image_path,
                              question=question,
                              answers=answer,
                              example_type=value["genre"])
                  examples.append(ex)
                  sid += 1
        print(f"{self.task_name}: add {len(examples)} examples in all, and dropped {drop_num} examples.")
        return examples

    def calc_scores(self, res_examples: List[Example], metrics: List[str]=['acc']) -> Dict:
        """ Calculate scores with specified metrics.
          Args:
            @examples:
            @metrics:
          Return:
            A result dict keyed by metrics names.
        """
        metrics_scores = {"all": {}}
        for name in metrics:
          metric_cls = Registry.get_metric_class(name)
          scores = metric_cls.calc_scores(res_examples, self.examples)
          metrics_scores["all"][name] = scores
        # for eval_type in self.all_eval_types:
        #   c_res_example = [ex for ex in res_examples if ex.example_type == eval_type]
        #   c_ans_example = [ex for ex in self.examples if ex.example_type == eval_type]
        #   metrics_scores[eval_type] = {}
        #   for name in metrics:
        #     metric_cls = Registry.get_metric_class(name)
        #     scores = metric_cls.calc_scores(c_res_example, c_ans_example)
        #     metrics_scores[eval_type][name] = scores
        return metrics_scores
