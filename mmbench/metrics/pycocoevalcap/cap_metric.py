__author__ = 'tylin'
from mmbench.metrics.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from mmbench.metrics.pycocoevalcap.bleu.bleu import Bleu
from mmbench.metrics.pycocoevalcap.meteor.meteor import Meteor
from mmbench.metrics.pycocoevalcap.rouge.rouge import Rouge
from mmbench.metrics.pycocoevalcap.cider.cider import Cider
from mmbench.metrics.pycocoevalcap.spice.spice import Spice
from typing import List, Dict
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
    def calc_scores(self, pred_res: dict, gt_res: dict) -> Dict:
        """ Calculate scores of Bleu, METEOR, ROUGE_L, CIDEr, SPICE.
          Args:
            @pred_res: a dict where each key corresponds to a list of captions.
            @gt_res: a dict where each key corresponds to a list of captions.
          Return:
            the calculated metric scores.
        """
        # =================================================
        # Set up scorers
        # =================================================
        print('caption tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gt_res)
        res = tokenizer.tokenize(pred_res)

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
        return self.eval

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




@Registry.register_metric('CIDEr')
class CIDErMetric(BaseMetric):
    def __init__(self) -> None:
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}


    @classmethod
    def calc_scores(self, pred_res: dict, gt_res: dict) -> Dict:
        """ Calculate scores of Bleu, METEOR, ROUGE_L, CIDEr, SPICE.
          Args:
            @pred_res: a dict where each key corresponds to a list of captions.
            @gt_res: a dict where each key corresponds to a list of captions.
          Return:
            the calculated metric scores.
        """
        # =================================================
        # Set up scorers
        # =================================================
        print('caption tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gt_res)
        res = tokenizer.tokenize(pred_res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            # (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
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
        return self.eval

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