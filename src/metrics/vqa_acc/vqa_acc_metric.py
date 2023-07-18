from typing import Any
from src.common.example import Example
from src.common.registry import Registry
from src.metrics.base_metric import BaseMetric
from src.metrics.vqa_acc.vqa import VQA
from src.metrics.vqa_acc.vqa_eval import VQAEval
from typing import List, Dict



@Registry.register_metric('vqa_acc')
class VqaAccMetric(BaseMetric):
    def __init__(self) -> None:
        pass


    @classmethod
    def calc_scores(self, res_examples: List[Example], question_file: str, annotation_file: str) -> Dict:
        """ Use official VQA evaluation script to report metrics.
          Args:
            @res_examples: a list of result examples instanced by `Example` class.
            @question_file: the official question file downloaded from VQAv2 portal.
            @annotation_file: the official annotation file downloaded from VQAv2 portal.
          Return:
            the calculated metric scores.
        """
        scores = {}

        # Formatting
        result = []
        for ex in res_examples:
            for ans in ex.answers:
                result.append({'question_id': ex.idx, 'answer': ans})
        
        
        vqa = VQA(annotation_file, question_file)
        vqa_result = vqa.loadRes(result, question_file)

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