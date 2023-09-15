""" The scorer of Relaxed Accuracy (RA).
"""
from typing import Any
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
from typing import List, Dict, Optional

@Registry.register_metric('relaxed_acc')
class RelaxedAccMetric(BaseMetric):
    def __init__(self) -> None:
        pass


    @classmethod
    def calc_scores(self, trues: str, preds: str, max_relative_change: float = 0.05) -> bool:
        """Calculates relaxed correctness.

        The correctness tolerates certain error ratio defined by max_relative_change.
        See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
        “Following Methani et al. (2020), we use a relaxed accuracy measure for the
        numeric answers to allow a minor inaccuracy that may result from the automatic
        data extraction process. We consider an answer to be correct if it is within
        5% of the gold answer. For non-numeric answers, we still need an exact match
        to consider an answer to be correct.”

        Args:
        trues: a list of target strings.
        preds: a list of predicted strings.
        max_relative_change: Maximum relative change.

        Returns:
          Average accuracy score with respect to the tolerance.
        """

        def _to_float(text: str) -> Optional[float]:
            try:
                if text.endswith('%'):
                    # Convert percentages to floats.
                    return float(text.rstrip('%')) / 100.0
                else:
                    return float(text)
            except ValueError:
                return None

        correct = 0.0
        for gt_str, pr_str in zip(trues, preds):

            prediction_float = _to_float(pr_str)
            target_float = _to_float(gt_str)
            if prediction_float is not None and target_float:
                relative_change = abs(prediction_float -
                                    target_float) / abs(target_float)
                correct += int(relative_change <= max_relative_change)
            else:
                correct += int(pr_str.lower() == gt_str.lower())
        return correct / (len(trues) + 1e-15)