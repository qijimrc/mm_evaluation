from typing import Dict
import string
from nltk.tokenize import sent_tokenize

from sat.helpers import print_rank0
from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('COM')
class COMTask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'COM'
        super().__init__(task_cfg, **kw_args)
    
    def calc_scores(self, args, result_df) -> Dict:
        import ipdb
        ipdb.set_trace()
        """ Parse answer to separated explaination and answer.
        """
        com_scorer = Registry.get_metric_class('COMScore')
        translate_table = dict((ord(char), None) for char in string.punctuation)

        pred_expl_ans, gt_expl_ans = [], []
        for gt_ans, preds, gts in zip(result_df["answer"], result_df["preds"], result_df['com_chains_dialogues']):
            pred_expl, pred_ans = '', ''
            if preds:
                sents = sent_tokenize(preds)
                pred_expl, pred_ans = ''.join(sents[:-1]), sents[-1]
            gt_expl_s = ''
            gt_expl_s = [' '.join(sent_tokenize(dia)[:-1]) for dia in gts['com_chains_dialogues']]

            pure_gtan = gt_ans.translate(translate_table).strip()
            pure_predan = pred_ans.translate(translate_table).strip()
            # if re.match(fr'.*?({answer})$', pred, re.IGNORECASE):
            if pure_gtan!='' and (pure_predan.lower().rfind(pure_gtan.lower()) + len(pure_gtan) == len(pure_predan)):
                pred_ans = gt_ans

            pred_expl_ans.append([pred_expl, pred_ans])
            gt_expl_ans.append([gt_expl_s[0], gt_ans])
        
        res_metrics = com_scorer.calc_scores(gt_expl_ans, pred_expl_ans)
        return res_metrics
