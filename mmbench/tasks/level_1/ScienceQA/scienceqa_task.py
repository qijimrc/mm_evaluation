from typing import Any, Dict, List
import os
import json

from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.tasks.base_task import BaseTask
from mmbench.utils.utils import is_chinese

PROMPT_EN = "Please choose the correct option for the above question from the following options: "
PROMPT_ZH = "请从以下几个选项中选出上述问题的正确答案："

@Registry.register_task('ScienceQA')
class ScienceQA(BaseTask):
    def __init__(self, task_cfg):

        self.task_name = 'ScienceQA'
        self.img_dir = task_cfg.img_dir
        self.anns_paths = task_cfg.anns_paths
        self.metrics = task_cfg.metrics
        self.all_eval_types = ["NAT", "SOC", "LAN", "TXT", "IMG", "NO", "G1-6", "G7-12"]
        
        super().__init__(self.img_dir, self.anns_paths)
        

    def generate_prompt_in_multi_choice(self, choices, question):
        prompt = question + "\n" + (PROMPT_ZH if is_chinese(question) else PROMPT_EN) + "\n"
        start_op = 'A'
        for item in choices:
            prompt += f'{start_op}: {item}\n'
            start_op = chr(ord(start_op) + 1)
        return prompt
      
    def _get_eval_type_in_subject(self, subject):
        topic_map = {
          "language science": "LAN",
          "natural science": "NAT",
          "social science": "SOC"
        }
        return [topic_map[subject]]
      
    def _get_eval_type_in_context(self, image, hint):
        ret = []
        if image is None and len(hint) <= 0:
          return ["NO"]
        if image:
          ret.append("IMG")
        if len(hint) > 0:
          ret.append("TXT")
        return ret
    
    def _get_eval_type_in_grade(self, grade):
        if grade in set(["grade1", "grade2", "grade3", "grade4", "grade5", "grade6"]):
            return ["G1-6"]
        if grade in set(["grade7", "grade8", "grade9", "grade10", "grade11", "grade12"]):
            return ["G7-12"]
        raise ValueError("Invalid grade: %s" % grade)

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
              if value["split"] != "test":
                continue
              image_path = None
              if value["image"]:
                  image_path = os.path.join(self.img_dir, key, value["image"])
                  if not os.path.exists(image_path):
                    print(f"image not found: {image_path}, will be skipped.")
                    drop_num += 1
                    continue
              # get example type list
              ex_types = []
              ex_types.extend(self._get_eval_type_in_subject(value["subject"]))
              ex_types.extend(self._get_eval_type_in_context(value["image"], value["hint"]))
              ex_types.extend(self._get_eval_type_in_grade(value["grade"]))
              # add example
              for ex_type in ex_types:
                  ex = Example(task=self.task_name,
                              idx=sid,
                              img_path=image_path,
                              question=self.generate_prompt_in_multi_choice(value["choices"], value["question"]),
                              answers=chr(ord('A') + value["answer"]),
                              example_type=ex_type,
                              context=value["hint"])
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
        metrics_scores = {}
        for eval_type in self.all_eval_types:
          c_res_example = [ex for ex in res_examples if ex.example_type == eval_type]
          c_ans_example = [ex for ex in self.examples if ex.example_type == eval_type]
          metrics_scores[eval_type] = {}
          for name in metrics:
            metric_cls = Registry.get_metric_class(name)
            scores = metric_cls.calc_scores(c_res_example, c_ans_example)
            metrics_scores[eval_type][name] = scores
        return metrics_scores
