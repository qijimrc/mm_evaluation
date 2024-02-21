from typing import List, Dict
from collections import defaultdict

from mmdoctor.metrics.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from mmdoctor.metrics.pycocoevalcap.bleu.bleu import Bleu
from mmdoctor.metrics.pycocoevalcap.meteor.meteor import Meteor
from mmdoctor.metrics.pycocoevalcap.rouge.rouge import Rouge
from mmdoctor.metrics.pycocoevalcap.cider.cider import Cider
from mmdoctor.metrics.pycocoevalcap.spice.spice import Spice
from mmdoctor.metrics.base_metric import BaseMetric
from mmdoctor.common.registry import Registry
from mmdoctor.common.logger import log

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
    def calc_scores(cls, results_data) -> Dict:
        """
        Args:
            results_data (list of dict): [
                {
                    "question_id": str (required),
                    "predict": str (required),
                    "answer": str (required),
                    "answer_list": str (optional, the priority is higher than answer)
                }, ...
            ]
        Returns:
            cls.imgToEval (dict) : {
                ...
            }
        """
        buffer_size = 4000 # The max length of string to pass into java
        # group results by image_id
        now = 0
        label_dict, pred_dict = {}, {}
        log(f"Cache caption scores with buffer_size {buffer_size}.")
        for i, c_data in enumerate(results_data):
            image_id = c_data["question_id"]
            label_dict[image_id] = c_data["answer_list"] if "answer_list" in c_data else [c_data["answer"]]
            pred_dict[image_id] = [c_data["predict"]]
            now += sum(map(len, label_dict[image_id])) + sum(map(len, pred_dict[image_id]))
            if now >= buffer_size:
                log(f"Cache {i}/{len(results_data)}.")
                metrics_scores = cls.compute_buffer(pred_dict, label_dict)
                label_dict, pred_dict = {}, {}
                now = 0
        if label_dict and pred_dict:
            metrics_scores = cls.compute_buffer(pred_dict, label_dict)
        log(f"Cache done.")
        all_scores = defaultdict(float)
        if "SPICE" in all_scores:
            all_scores['SPICE'] = {'pr': 0.0, 're': 0.0, 'f': 0.0, 'fn': 0.0, 'numImages': 0.0, 'fp': 0.0, 'tp': 0.0}
        for k in metrics_scores:
            for m in metrics_scores[k]:
                if m == 'image_id':
                    continue
                if m == 'SPICE':
                    for p in all_scores[m]:
                        all_scores[m][p] += metrics_scores[k][m]['All'][p]
                else:
                    all_scores[m] += metrics_scores[k][m]
        for m in all_scores:
            if type(all_scores[m]) is not dict:
                all_scores[m] /= len(metrics_scores)
            else:
                for p in all_scores[m]:
                    all_scores[m][p] /= len(metrics_scores)
        cls.empty_cache()
        return all_scores

    @classmethod
    def compute_buffer(cls, pred_res: dict, gt_res: dict) -> Dict:
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
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gt_res)
        res = tokenizer.tokenize(pred_res)

        # =================================================
        # Set up scorers
        # =================================================
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    cls.setEval(sc, m)
                    cls.setImgToEvalImgs(scs, gts.keys(), m)
            else:
                cls.setEval(score, method)
                cls.setImgToEvalImgs(scores, gts.keys(), method)
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
    res = CaptionMetric.compute_buffer(pred, gt)
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
    res = CaptionMetric.compute_buffer(pred, gt)