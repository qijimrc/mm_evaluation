import re
from sklearn.metrics import accuracy_score, ndcg_score
import textdistance
from nltk.translate.bleu_score import sentence_bleu

from mmbench.common.example import Example
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
from typing import List, Dict, Tuple

@Registry.register_metric('COMScore')
class COMScore(BaseMetric):
    def __init__(self) -> None:
        self.manipulations = [
            'GROUNDING', 'OCR', 'CROP_AND_ZOOMIN', 'CALCULATE'
        ]

    def _normalize_frags(cls, bbxs_str_list):
        res = []
        for bbxs_str in bbxs_str_list:
            try:
                bbxs_str = str(eval(bbxs_str))
            except:
                pass
            res.append(bbxs_str)
        return res
    
    def _frag2ids(cls, frags1, frags2):
        vocab = {fr:i for i,fr in enumerate(set(frags1).union(frags2))}
        # seq1 = [ord('0')+vocab[fr] for fr in frags1]
        # seq2 = [ord('0')+vocab[fr] for fr in frags2]
        seq1 = [vocab[fr] for fr in frags1]
        seq2 = [vocab[fr] for fr in frags2]
        return seq1, seq2
                

    @classmethod
    def calc_scores(cls, trues: List[Tuple[str, str]], preds: List[Tuple[str, str]], calc_explain=True) -> Dict:
        """ Calculate scores for the generated Chain of Manipulations.
          Args:
            @trues: a list of golden tuples, each contains a pair of an explanation and a answer.
            @preds: a list of predicted tuples, each contains a pair of an explanation and a answer. (the order should match the `trues`)
            @calc_explain: whether to calculate the score for the correctness of explanation. (using NDCG for calculation)

                acc_explain(expl1, expl2) = NDCG({funcs, boxes}_expl1, {funcs, boxes}_expl2) * BLEU(expl_1, expl_2)
        """
        assert len(trues) == len(preds)
        if len(trues) == 0:
            return 0.0
        
        explain_score, explain_sum = 0.0, 0.0
        if calc_explain:
            ndcg_ptr = re.compile('({MPS}|\[[\[\]\d\,\s]{0,100}\])'.format(MPS='|'.join(cls.manipulations)))
            for (expl_g, ans_g), (expl_p, ans_p)  in zip(trues, preds):
                frags_g = cls._normalize_frags(ndcg_ptr.findall(expl_g))
                frags_p = cls._normalize_frags(ndcg_ptr.findall(expl_p))
                seq1, seq2 = cls._frag2ids(frags_g, frags_p)
                score_frags = textdistance.levenshtein(seq1, seq2) / max(len(seq1), len(seq2))
                score_txt = sentence_bleu([expl_g.split()], expl_p.split())
                explain_sum += score_frags * score_txt
            explain_score = explain_sum / len(trues)

        ans_score = 0.0
        true_anss, pred_anss = [], []
        for (expl_g, ans_g), (expl_p, ans_p)  in zip(trues, preds):
            true_anss.append(ans_g)
            pred_anss.append(ans_p)
        ans_score = accuracy_score(trues, preds)

        return {'explain_score': explain_score, 'ans_score':ans_score}
