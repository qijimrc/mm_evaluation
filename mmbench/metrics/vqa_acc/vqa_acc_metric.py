from typing import Any
from mmbench.common.example import Example
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
from mmbench.metrics.vqa_acc.vqa import VQA
from mmbench.metrics.vqa_acc.vqa_eval import VQAEval
from typing import List, Dict
import re



@Registry.register_metric('vqa_acc')
class VqaAccMetric(BaseMetric):
    def __init__(self) -> None:
        pass


    @classmethod
    def calc_scores(self, res_examples: List[Example], gt_examples: List[Example], vqav2_info: Dict) -> Dict:
        """ Use official VQA evaluation script to report metrics.
          Args:
            @res_examples: a list of result examples instanced by `Example` class.
            @gt_examples: a list of ground truth examples.
          Return:
            the calculated metric scores.
        """
        scores = {}

        # Formatting
        result = []
        for ex in res_examples:
            for ans in ex.answers:
                result.append({'question_id': ex.idx, 'answer': ans})

        vqav2_info['questions'] = []
        vqav2_info['annotations'] = []
        for ex in gt_examples:
            vqav2_info['questions'].append({
                'image_id': re.match(r'.*?image_id=(\d+).*', ex.context).group(1),
                'question_id': ex.idx,
                'question': ex.question,
                'question_type': ex.example_type,
            })
            vqav2_info['annotations'].append({
                'image_id': ex.example_type,
                'question_id': ex.idx,
                'question': ex.question,
                'question_type': ex.example_type,
                'answers': [{'answer': a, 'answer_confidence':'yes', 'answer_id':_i} for _i,a in enumerate(ex.answers)],
                'answer_type': 'other',
            })
        
        
        vqa = VQA(annotation_dict=vqav2_info)
        vqa_result = vqa.loadRes(resDict=result, annotation_dict=vqav2_info)

        # create vqaEval object by taking vqa and vqaRes
        # n is precision of accuracy (number of places after decimal), default is 2
        vqa_scorer = VQAEval(vqa, vqa_result, n=2)
        print("Start VQA evaluation.")
        vqa_scorer.evaluate()

        # print accuracies
        overall_acc = vqa_scorer.accuracy["overall"]
        scores["agg_metrics"] = overall_acc

        print("Overall Accuracy is: %.02f\n" % overall_acc)
        print("Per Answer Type Accuracy is the following:")

        for ans_type in vqa_scorer.accuracy["perAnswerType"]:
            print(
                "%s : %.02f"
                % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
            )
            scores[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

        # with open(
        #     os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        # ) as f:
        #     f.write(json.dumps(metrics) + "\n")

        return scores