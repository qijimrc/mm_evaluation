from typing import Any

class BaseMetric:
    def __init__(self) -> None:
        pass

    @NotImplementedError
    @classmethod
    def calc_scores(self, *args: Any, **kwds: Any) -> Any:
        """Calculate metric scores of model predictions against the ground truths.
        Args:
        Return:
            a scores dict keyed by metric names / numeric / ...
        """
        pass
    