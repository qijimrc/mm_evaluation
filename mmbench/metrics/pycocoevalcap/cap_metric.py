__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
from typing import Any
from typing import List, Dict
import collections
from mmbench.common.example import Example
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric



@Registry.register_metric('caption')
class CaptionMetric(BaseMetric):
    def __init__(self) -> None:
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}


    @classmethod
    def calc_scores(self, res_examples: List[Example], gt_examples: List[Example]) -> Dict:
        """ Use official VQA evaluation script to report metrics.
          Args:
            @res_examples: a list of result examples instanced by `Example` class.
            @gt_examples: a list of ground truth examples.
          Return:
            the calculated metric scores.
        """
        scores = {}

        gts = collections.defaultdict(list)
        res = collections.defaultdict(list)
        for ex in gt_examples:
            gts[ex.idx].append(ex.answers)
        for ex in res_examples:
            res[ex.idx].append(ex.answers)

        # =================================================
        # Set up scorers
        # =================================================
        print('caption tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]