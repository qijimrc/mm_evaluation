import os
import jsonlines
from typing import Dict, List
import pandas as pd

from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.tasks.base_task import BaseTask
from mmbench.utils.utils import is_chinese

PROMPT_EN = "Please choose the correct option for the above question from the following options: "
PROMPT_ZH = "请从以下几个选项中选出上述问题的正确答案："

@Registry.register_task('HalVQA')
class HalVQATask(BaseTask):
    def __init__(self, task_cfg):

        self.task_name = 'HalVQA'
        self.img_dir = task_cfg.img_dir
        self.anns_paths = task_cfg.anns_paths
        self.metrics = task_cfg.metrics
        self.example_types = ["existence", "color", "position"]

        super().__init__(self.img_dir, self.anns_paths)

    def generate_prompt_in_multi_choice(self, choices, question):
        prompt = question + "\n" + (PROMPT_ZH if is_chinese(question) else PROMPT_EN) + "\n"
        start_op = 'A'
        for item in choices:
            prompt += f'{start_op}: {item}\n'
            start_op = chr(ord(start_op) + 1)
        return prompt

    def to_examples(self, img_dir: str, anns_paths: List) ->List[Example]:
        """ Convert annotations to canonical examples.
          Args:
            @img_dir: the root dir of vision source.
            @anns_paths: the paths of annotation files.
          Return:
            A list of examples instanced from the `Example` class.
        """
        examples = []
        
        drop_num = 0
        with jsonlines.open(anns_paths, 'r') as fp:
            sid = 0
            for value in fp:
                image_path = os.path.join(self.img_dir, value["image"])
                if not os.path.exists(image_path):
                  print(f"image not found: {image_path}, will be skipped.")
                  drop_num += 1
                  continue
                assert value["eval_type"] in self.example_types
                # get example type list
                ex = Example(task=self.task_name,
                            idx=sid,
                            img_path=image_path,
                            question=self.generate_prompt_in_multi_choice(value["choices"], value["question"]),
                            answers=chr(ord('A') + value["choices"].index(value["answer"])),
                            example_type=value["eval_type"])
                examples.append(ex)
                sid += 1
        print(f"{self.task_name}: add {len(examples)} examples in all, and dropped {drop_num} examples.")
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
        for ext in self.example_types:
            cur_res_examples = [ex for ex in res_examples if ex.example_type == ext]
            if len(cur_res_examples) > 0:
              cur_ans_examples = [ex for ex in self.examples if ex.example_type == ext]
              metrics_scores[ext] = {}
              for name in metrics:
                metric_cls = Registry.get_metric_class(name)
                scores = metric_cls.calc_scores(cur_res_examples, cur_ans_examples)
                metrics_scores[ext][name] = scores
        return metrics_scores
        