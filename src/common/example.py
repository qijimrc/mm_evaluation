from typing import Dict, List, Any
import json


class Example:
    """ An evaluation example.
    """
    def __init__(self,
                 task: str,
                 idx: int=0,
                 vis: str=None,
                 context: Any=None,
                 question: str=None,
                 answers: str=None) -> None:
        """ Construct an example.
          Args:
            @task: the task name.
            @idx: the index of current example.
            @vis: the path of vison input (e.g., image).
            @context: the context.
            @question: the language question.
            @answer: a list of language answers.
        """
        self.task = task
        self.idx = idx
        self.vis = vis
        self.context = context
        self.question = question
        self.answers = answers


    def to_json(self,):
      return {'task': self.task, 'idx':self.idx, 'vis':self.vis,
                'context':self.context, 'question':self.question, 'answers':self.answers}
    
    @classmethod
    def from_json(example_js: dict):
       """ Initiate class from json object.
       """
       return Example(**example_js)
    
    def __str__(self) -> str:
      return json.dumps(self.to_json())
