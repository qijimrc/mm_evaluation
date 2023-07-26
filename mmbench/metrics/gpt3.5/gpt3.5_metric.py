""" Use GPT3.5 to evaluate the MLLM's prediction performance.
"""

from typing import Any
from mmbench.common.example import Example
from mmbench.common.registry import Registry
from mmbench.metrics.base_metric import BaseMetric
import re
from typing import List, Dict
import openai


openai.proxy = {
            # "http": "http://127.0.0.1:8002",
            # "https": "http://127.0.0.1:8002",
            'http': 'http://127.0.0.1:1087',
            'https': 'http://127.0.0.1:1087'
        }
openai.api_key = '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'




@Registry.register_metric('gpt35_metric')
class GPT35Metric(BaseMetric):
    def __init__(self) -> None:
        
        
        # Initializing prepends prompt
        self.instruction = [
            {'role': 'System'},
            {'content': 'You are a reviewer for a multimodal model. Given an image with the task instruction, a question and the golden answer for the image, you are able to score the asnwer predicted by the model, which is the degree to which the predicted answer match the correct answer out of 100. The image information will be given to you in the form of an caption and a scene graph, where the scene graph contains the objects in the image, the coordinates of the objects and the spatial relationships between the objects.'}
        ]

        self.demonstrations = [
            {'role': 'User'},
            {'content': 'Caption: a person in a boat. Scene Graph: [(jacket:324.0,141.0,377.0,216.0), (man:316.0,117.0,383.0,226.0), (person:306.0,116.0,380.0,227.0), (man,wearing,jacket), (person,wearing,jacket), (hill-1,behind,boat), (mountain,behind,boat)]. Question: What is in front of the boat? Answer: a mountain. Prediction: There is a mountain in front of the boat.'},
            {'role': 'Assistant'},
            {'content': '100'}
        ]

        self.prepend_prompt = self.instruction + self.demonstrations


    @classmethod
    def calc_scores(self, res_examples: List[Example], gold_examples: List[Example]) -> Dict:
        """ Use official VQA evaluation script to report metrics.
          Args:
            @res_examples: a list of result examples instanced by `Example` class.
          Return:
            the calculated scores.
        """
        scores = {}

        gold_examples = {ex.idx: ex for ex in gold_examples}

        gpt_scores = []
        for ex in res_examples:

            prompt = [
                {'User': 
                 'Caption: {caption}. Scene Graph: {sg}. Question: {question}, Answer: {anser}, Prediction: {prediction}'.format(
                    caption= '',
                    sg = '',
                    question = ex.question,
                    answer = gold_examples[ex.idx],
                    prediction = ex.answers
                 )
                }
            ]

            response = openai.ChatCompletion.create(
                # model='text-davinci-002',
                model='gpt-3.5-turbo',
                messages = self.prepend_prompt + prompt
            )

            gpt_scores.append(float(re.match(r'^(\d+).*', response).group(1)))
        
        scores['gpt35_scores'] = gpt_scores

        return scores