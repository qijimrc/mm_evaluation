import os, json
from typing import Any
from mmdoctor.common.registry import Registry
from mmdoctor.common.logger import log
from mmdoctor.metrics.base_metric import BaseMetric

question_ids_to_exclude = []

@Registry.register_metric('ANLS')
class ANLSScorer(BaseMetric):
    def __init__(self) -> None:
        pass

    @classmethod
    def levenshtein_distance(cls, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    @classmethod
    def calc_scores(cls, results_data):
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
            ResDict (dict) : {
                "final_score": float,
                ...
            }
        """
        label_list, pred_list = [], []
        for c_data in results_data:
            question_id = c_data["question_id"]
            if "answer_list" not in c_data or len(c_data["answer_list"]) == 0:
                c_data["answer_list"] = [c_data["answer"]]
            label_list.append({"QuestionId": question_id, "answers": c_data["answer_list"]})
            pred_list.append({"QuestionId": question_id, "answer": c_data["predict"]})
        return cls.compute(pred_list, label_list)

    @classmethod
    def compute(cls, pred_qas, gt_qas, anls_threshold=0.5, answer_types=False, output_persample_scores=False):
        """Claculate the Average Normalized Levenshtein Similarity (ANLS). The length of pred_qas must equals to the length of gt_qas.
          Args:
            @pred_qas [{'QuestionId': qid, 'answer': ans}]: a list of dict where each contains keys of `QuestionId` and `answer`.
            @gt_qas [{'QuestionId':qid, 'answers': ans_list}]: a list dict where each contains keys of `QuestionId` and `answers`. 
          Return:
            metric score.
        """
        if set([ex['QuestionId'] for ex in pred_qas]) != set([ex['QuestionId'] for ex in gt_qas]):
            log(f"The {len(pred_qas)} of predictions does not equal to the {len(gt_qas)} of goldens.")
        
        show_scores_per_answer_type = answer_types
        perSampleMetrics = {}
        predJson = {ex['QuestionId']:ex for ex in pred_qas}
        
        totalScore = 0
        row = 0
        
        if show_scores_per_answer_type:
            answerTypeTotalScore = {x:0 for x in answer_types.keys()}
            answerTypeNumQuestions = {x:0 for x in answer_types.keys()}

        for gtObject in gt_qas:

            q_id = gtObject['QuestionId'];
            detObject = predJson[q_id];

            if q_id in question_ids_to_exclude:
                question_result = 0
                info = 'Question EXCLUDED from the result'
            else:
                info = ''
                values = []
                for answer in gtObject['answers']:
                    # preprocess both the answers - gt and prediction
                    gt_answer = ' '.join(answer.strip().lower().split())
                    det_answer = ' '.join(detObject['answer'].strip().lower().split())
                    #dist = levenshtein_distance(answer.lower(), detObject['answer'].lower())
                    dist = cls.levenshtein_distance(gt_answer,det_answer)
                    length = max(len(answer.upper()), len(detObject['answer'].upper()))
                    values.append(0.0 if length == 0 else float(dist) / float(length))

                question_result = 1 - min(values)
            
                if (question_result < anls_threshold) :
                    question_result = 0

                totalScore += question_result
                
                if show_scores_per_answer_type:
                    for q_type in gtObject["answer_type"]:
                        answerTypeTotalScore[q_type] += question_result
                        answerTypeNumQuestions[q_type] += 1
            
            perSampleMetrics[str(gtObject['QuestionId'])] = {
                                    'score':question_result,
                                    'gt':gtObject['answers'],
                                    'det':detObject['answer'],
                                    'info': info
                                    }
            row = row + 1
                                    
        resDict = {
            'final_score': 0 if len(gt_qas) == 0 else totalScore/ (len(gt_qas) - len(question_ids_to_exclude))
        }

        if show_scores_per_answer_type:
            answer_types_scores = {}
            for a_type, ref in answer_types.items():
                answer_types_scores[ref] = 0 if len(gt_qas) == 0 else answerTypeTotalScore[a_type] / (answerTypeNumQuestions[a_type])
            resDict["scores_by_types"] =  answer_types_scores

        if output_persample_scores:
            resDict["per_sample_result"] = perSampleMetrics

        return resDict