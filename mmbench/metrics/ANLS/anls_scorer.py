import os, json
from typing import Any
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric

question_ids_to_exclude = []


@Registry.register_metric('ANLS')
class ANLSScorer(BaseMetric):
    def __init__(self) -> None:
        # self.answer_types = {'image span': 'Image-Span', 'question span': 'Question-Span', 'multiple spans': 'Multi-Span', 'non span': 'None span', 'list': 'List'}
        self.answer_types = {'image span': 'Image-Span', 'question span': 'Question-Span', 'multiple spans': 'Multi-Span', 'non span': 'None span'}
        self.evidence_types = {'table/list': 'Table/list', 'textual': 'Text', 'photo/pciture/visual_objects': 'Visual/Layout', 'figure': 'Figure', 'map': 'Map'}
        self.reasoning_requirements = {'comparison': 'Sorting', 'arithmetic': 'Arithmetic', 'counting':'Counting'}

    @classmethod
    def levenshtein_distance(self, s1, s2):
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
    def calc_scores(self, pred_qas, gt_qas, anls_threshold=0.5, answer_types=False, output=None):
        """ Claculate the Average Normalized Levenshtein Similarity (ANLS). The length of pred_qas must equals to the length of gt_qas.
          Args:
            @pred_qas [{'QuestionId': qid, 'answer': ans}]: a list of dict where each contains keys of `QuestionId` and `answer`.
            @gt_qas [{'QuestionId':qid, 'answers': ans_list}]: a list dict where each contains keys of `QuestionId` and `answers`. 
          Return:
            metric score.
        """  
        if set([ex['QuestionId'] for ex in pred_qas]) != set([ex['QuestionId'] for ex in gt_qas]):
            print(f"The {len(pred_qas)} of predictions does not equal to the {len(gt_qas)} of goldens.")
        
        show_scores_per_answer_type = answer_types
        perSampleMetrics = {}
        res_id_to_index = {int(r['questionId']): ix for ix, r in enumerate(pred_qas)}
        predJson = {ex['QuestionId']:ex for ex in pred_qas}
        
        totalScore = 0
        row = 0
        
        if show_scores_per_answer_type:
            answerTypeTotalScore = {x:0 for x in answer_types.keys()}
            answerTypeNumQuestions = {x:0 for x in answer_types.keys()}

            evidenceTypeTotalScore = {x:0 for x in self.evidence_types.keys()}
            evidenceTypeNumQuestions = {x:0 for x in self.evidence_types.keys()}

            reasoningTypeTotalScore = {x:0 for x in self.reasoning_requirements.keys()}
            reasoningTypeNumQuestions = {x:0 for x in self.reasoning_requirements.keys()}
        
        for gtObject in gt_qas:

            q_id = int(gtObject['questionId']);
            res_ix = res_id_to_index[q_id];
            detObject = predJson[res_ix];

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
                    dist = self.levenshtein_distance(gt_answer,det_answer)
                    length = max( len(answer.upper()), len(detObject['answer'].upper()) )
                    values.append( 0.0 if length == 0 else float(dist) / float(length) )

                question_result = 1 - min(values)
            
                if (question_result < self.evaluationParams.anls_threshold) :
                    question_result = 0

                totalScore += question_result
                
                if show_scores_per_answer_type:
                    for q_type in gtObject["answer_type"]:
                        answerTypeTotalScore[q_type] += question_result
                        answerTypeNumQuestions[q_type] += 1

                    for q_type in gtObject["evidence"]:
                        evidenceTypeTotalScore[q_type] += question_result
                        evidenceTypeNumQuestions[q_type] += 1

                    for q_type in gtObject["operation/reasoning"]:
                        reasoningTypeTotalScore[q_type] += question_result
                        reasoningTypeNumQuestions[q_type] += 1
                    
            
            perSampleMetrics[str(gtObject['questionId'])] = {
                                    'score':question_result,
                                    'question':gtObject['question'],
                                    'gt':gtObject['answers'],
                                    'det':detObject['answer'],
                                    'info': info
                                    }
            row = row + 1

                                    
        methodMetrics = {
            'score': 0 if len(gt_qas) == 0 else totalScore/ (len(gt_qas) - len(question_ids_to_exclude) )
        }

        answer_types_scores = {}
        evidence_types_scores = {}
        operation_types_scores = {}

        if show_scores_per_answer_type:
            for a_type, ref in answer_types.items():
                answer_types_scores[ref] = 0 if len(gt_qas) == 0 else answerTypeTotalScore[a_type] / (answerTypeNumQuestions[a_type] )

            for e_type, ref in self.evidence_types.items():
                evidence_types_scores[ref] = 0 if len(gt_qas) == 0 else evidenceTypeTotalScore[e_type] / (evidenceTypeNumQuestions[e_type] )

            for r_type, ref in self.reasoning_requirements.items():
                operation_types_scores[ref] = 0 if len(gt_qas) == 0 else reasoningTypeTotalScore[r_type] / (reasoningTypeNumQuestions[r_type] )


        resDict = {
                'result': methodMetrics, 
                'scores_by_types': {'answer_types': answer_types_scores, 'evidence_types': evidence_types_scores, 'operation_types': operation_types_scores},
                'per_sample_result':perSampleMetrics
                }

        return resDict