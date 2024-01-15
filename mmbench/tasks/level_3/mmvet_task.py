from typing import Dict
import string
from nltk.tokenize import sent_tokenize

from sat.helpers import print_rank0
from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('MMVet')
class MMVetTask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'MMVet'
        super().__init__(task_cfg, **kw_args)
    
    def calc_scores(self, args, result_df) -> Dict:
        metrics = {}
        return metrics
