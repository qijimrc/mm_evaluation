from typing import Dict, List, Any
import json


class Example:
    """ An evaluation example.
    """
    def __init__(self,
                 task: str,
                 img_path: str,
                 question: str,
                 answers: List,
                 idx: int=0,
                 context: Any=None,
                 instruction: str= None) -> None:
        """ Construct an example.
          Args:
            @task: the task name.
            @idx: the index of current example.
            @img_path: the path of vison input (e.g., image).
            @context: the context.
            @question: the language question.
            @answer: a list of language answers.
        """
        self.idx = idx
        self.task = task
        self.instruction = instruction
        self.img_path = img_path
        self.context = context
        self.question = question
        self.answers = answers


    def to_json(self,):
      return {'task': self.task, 'img_path':self.img_path, 'question':self.question, 'answers':self.answers,
              'idx':self.idx, 'context':self.context, 'instruction': self.instruction}
    
    @classmethod
    def from_json(example_js: dict):
       """ Initiate class from json object.
       """
       return Example(**example_js)
    
    def __str__(self) -> str:
      return json.dumps(self.to_json())
