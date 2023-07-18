from src.common.registry import Registry
from src.common.example import Example
from typing import Any, Dict, List, Optional
from torch.utils.data.dataset import Dataset

class BaseTask:
    def __init__(self, vis_root: str, anns_paths: List[str]):

        self.vis_root = vis_root
        self.anns_paths = anns_paths

        self.examples = self.to_examples(vis_root, anns_paths)
        self.cur = 0


    def __call__(self, index=None) -> Example:
        """ Default iterator to return current example of given index.
        """
        if index:
            if index >= len(self.examples): raise StopIteration
            ex = self.examples[index]
        else:
            if self.cur >= len(self.examples): raise StopIteration
            ex = self.examples[self.cur]
            self.cur += 1
        return ex

    @NotImplementedError
    def to_examples(self, vis_root: str, anns_paths: List) ->List[Example]:
        """ Convert annotations to canonical examples.
          Args:
            @vis_root: the root dir of vision source.
            @anns_paths: the paths of annotation files.
          Return:
            A list of examples instanced from the `Example` class.
        """
        pass

    @NotImplementedError
    def calc_scores(self, res_examples: List[Example], metrics: List[str]) -> Dict:
        """ Calculate scores with specified metrics.
          Args:
            @res_examples: a list of prediction examples instanced from `Example` class. 
            @metrics: the metric names to be evaluated.
          Return:
            A result dict keyed by metrics names.
        """
        pass