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
                 answer: str=None) -> None:
        """ Construct an example.
          Args:
            @task: the task name.
            @idx: the index of current example.
            @vis: the path of vison input (e.g., image).
            @context: the context.
            @question: the language question.
            @answer: the language answer.
        """
        self.task = task
        self.idx = idx
        self.vis = vis
        self.context = context
        self.question = question
        self.answer = answer

    def __str__(self) -> str:
        return json.dumps({'task': self.task, 'idx':self.idx, 'vis':self.vis,
                'context':self.context, 'question':self.question, 'answer':self.answer})