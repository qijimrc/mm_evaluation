import sys, os
from typing import Any
sys.path.append('/data/qiji/repos/LAVIS')
sys.path.append('/data/qiji/repos/mm_evaluation')
from src.evaluator import Evaluator
from src.common.example import Example
import torch
from lavis.models import load_model, load_model_and_preprocess


class ModelInterface:
    def __init__(self, device: str='cuda:3') -> None:
        # setup device to use
        self.device = torch.device(device)

        # we associate a model with its preprocessors to make it easier for inference.
        # self.model, self.vis_processors, _ = load_model_and_preprocess(
        #     name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
        # )

        # Other available models:
        # 
        # model, vis_processors, _ = load_model_and_preprocess(
        #     name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
        # )
        # model, vis_processors, _ = load_model_and_preprocess(
        #     name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
        # )
        # model, vis_processors, _ = load_model_and_preprocess(
        #     name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
        # )
        # model, vis_processors, _ = load_model_and_preprocess(
        #     name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
        # )
        #
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )
        #
        # model, vis_processors, _ = load_model_and_preprocess(
        #     name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
        # )

        # vis_processors.keys()
        torch.cuda.set_device(device)

    def __call__(self, example) -> str:
        img = self.vis_processors['eval'](example.vis).unsqueeze(0).to(device)
        out = self.model.generate({
                'image': img,
                'prompt': "Question: %s" % example.question,
        })
        return out



def evaluating_in_code(model_interface):
    """ Use our benchmark in your training code.
    """
    import ipdb
    ipdb.set_trace()
    # Initialize evaluator with all tasks
    evaluator = Evaluator()

    predictions = []
    for ex in evaluator.get_mixed_dataloader():
        ans = model_interface(ex)
        predictions.append(Example(task=ex.task, idx=ex.idx, answers=[ans]))
    metrics_scores = evaluator.evaluate_examples(predictions)

    

def evaluating_on_results(model_interface, save_js='results.json'):
    """ Use our benchmark in your training code.
    """
    predictions = []
    for ex in evaluator.get_mixed_dataloader():
        ans = model_interface(ex)
        predictions.append(Example(task=ex.task, idx=ex.idx, answers=[ans]))
    
    # Save the result for analysing and evaluating later
    evaluator.save_canonical_results(predictions)

    # Initialize evaluator with all tasks
    evaluator = Evaluator()
    evaluator.evaluate_files(save_js)

if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

    model_interface = ModelInterface()

    evaluating_in_code(model_interface)

    evaluating_on_results(model_interface, save_js='blip2_result.json')