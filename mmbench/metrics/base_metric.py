from mmbench.common.example import Example
from typing import Any, List, Dict

class BaseMetric:
    def __init__(self) -> None:
        pass


    @NotImplementedError
    @classmethod
    def calc_scores(self, *args: Any, **kwds: Any) -> Any:
    # def calc_scores(self, res_examples: List[Example], gts_examples: List[Example]) -> Dict:
        """ calculate metric scores of model predictions against the ground truths.
          Args:
          Return:
            a scores dict keyed by metric names.
        """
        pass
    