import os
import json
import random
from functools import wraps

from mmdoctor.common.utils import is_chinese

class BaseDataset(object):
    def __init__(self, args, custom_data_hooks=dict()) -> None:
        """
        Args:
            args (dict): args
            custom_data_hooks (dict, optional): customed data processors
                {
                    "datatype-name": function as
                        def [datatype-name](metadata, uni_key, img_path, **kwargs):
                            '''Data Processor

                            Args:
                                metadata (dict): normalized data
                                uni_key (str): unique key
                                img_path (str): image path
                            
                            Return: a tuple of (question, history)
                                question (str): query
                                history (list): [(q1, a1), (q2, a2), ...]
                            '''
                            pass
                }
        """
        self.args = args
        self.custom_data_hooks = custom_data_hooks

        self.img_pad = os.path.join(os.path.dirname(__file__), "assets/no_img.png")
        with open(os.path.join(os.path.dirname(__file__), "assets/templates_en.json"), "r") as fp:
            self.templates_en = json.load(fp)
        with open(os.path.join(os.path.dirname(__file__), "assets/templates_zh.json"), "r", encoding='utf-8') as fp:
            self.templates_zh = json.load(fp)
    
    def normal_qa(self, metadata, uni_key, **kwargs):
        return metadata["question"], []

    def normal_caption(self, metadata, uni_key, **kwargs):
        language_zh = is_chinese(metadata["answer"])
        template = self.templates_zh if language_zh else self.templates_en
        prompt = random.choice(template["Caption"]).replace('<image>', '')
        return prompt, []

    def multichoice(self, metadata, uni_key, **kwargs):
        choices, question = metadata["choices"], metadata["question"]
        language_zh = is_chinese(question)
        template = self.templates_zh if language_zh else self.templates_en
        choice_prompt = template["Choices"]
        choice_prompt = random.choice(choice_prompt)
        prompt = question + "\n" + choice_prompt + "\n"
        start_op = 'A'
        for item in choices:
            prompt += f'{start_op}: {item}\n'
            start_op = chr(ord(start_op) + 1)
        prompt += "回答: " if language_zh else "Answer:"
        return prompt, []
    
    def multi_vqa(self, metadata, uni_key, **kwargs):
        assert len(metadata["question"]) == len(metadata["answer"]), \
            f"[{uni_key}]: question and answer should have the same length, but got {len(metadata['question'])} and {len(metadata['answer'])}"
        vqa_length = len(metadata["question"])
        assert vqa_length > 0, "[%s]: question and answer should not be empty" % uni_key
        history = []
        for i in range(vqa_length-1):
            history.append((metadata["question"][i], metadata["answer"][i]))
        return metadata["question"][-1], history
    