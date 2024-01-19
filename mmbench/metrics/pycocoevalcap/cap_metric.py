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
    evalImgs = []
    eval = {}
    imgToEval = {}
    
    @classmethod
    def empty_cache(cls):
        cls.evalImgs = []
        cls.eval = {}
        cls.imgToEval = {}

    @classmethod
    def calc_scores(cls, pred_res: dict, gt_res: dict) -> Dict:
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
                    cls.setEval(sc, m)
                    cls.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                cls.setEval(score, method)
                cls.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        cls.setEvalImgs()
        return cls.imgToEval

    @classmethod
    def setEval(cls, score, method):
        cls.eval[method] = score

    @classmethod
    def setImgToEvalImgs(cls, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in cls.imgToEval:
                cls.imgToEval[imgId] = {}
                cls.imgToEval[imgId]["image_id"] = imgId
            cls.imgToEval[imgId][method] = score

    @classmethod
    def setEvalImgs(cls):
        cls.evalImgs = [eval for imgId, eval in cls.imgToEval.items()]

if __name__ == '__main__':
    pred = {
        1: ["hello, how are you?"],
        2: ["I'm fine"],
        3: ["ok"],
    }
    gt = {
        1: ["hello"],
        2: ["I'm fine"],
        3: ["ok"],
    }
    res = CaptionMetric.calc_scores(pred, gt)
    pred = {
        4: ["hello, how are you?"],
        5: ["I'm fine"],
        6: ["ok"],
    }
    gt = {
        4: ["hello"],
        5: ["I'm fine"],
        6: ["ok"],
    }
    res = CaptionMetric.calc_scores(pred, gt)

    breakpoint()