from src.common.registry import Registry
from src.common.example import Example
from src.tasks.base_task import BaseTask
import json
from typing import Any, Dict, List



@Registry.register_task('VQAv2')
class VQAv2Task(BaseTask):
    def __init__(self, ):

        self.vis_root = ''
        self.anns_paths = [
            '',
            ''
        ]
        self.metrics = ['']
        self.examples = []

        super().__init__(self.vis_root, self.anns_paths)

    def to_examples(self, vis_root: str, anns_paths: List) ->List[Example]:
        """ Convert annotations to canonical examples.
          Args:
            @vis_root: the root dir of vision source.
            @anns_paths: the paths of annotation files.
          Return:
            A list of examples instanced from the `Example` class.
        """
        pass

    def calc_scores(self, examples: List[Example], metrics: List[str]) -> Dict:
        """ Calculate scores with specified metrics.
          Args:
            @examples:
            @metrics:
          Return:
            A result dict keyed by metrics names.
        """
        pred_exs, gold_exs = examples, self.examples
        


