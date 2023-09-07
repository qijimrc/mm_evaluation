import os
import json
import random

from sat.helpers import print_rank0
from mmbench.common.utils import is_chinese

class BaseDataset(object):
    def __init__(self, mt, args, data_mode, other_attr=[]) -> None:
        self.mt = mt
        self.args = args
        self.data_mode = data_mode
        self.other_attr = other_attr

        self.img_pad = os.path.join(os.path.dirname(__file__), "assets/no_img.png")
        with open(os.path.join(os.path.dirname(__file__), "assets/templates_en.json"), "r") as fp:
            self.templates_en = json.load(fp)
        with open(os.path.join(os.path.dirname(__file__), "assets/templates_zh.json"), "r") as fp:
            self.templates_zh = json.load(fp)

    def process_img(self, img):
        img_dict = {'vision': self.mt.image_processor(img)}
        if self.mt.cross_image_processor:
            img_dict.update({'cross': self.mt.cross_image_processor(img)})
        return img_dict
    
    def process_text(self, prompt, answer):
        return self.mt.text_processor(answer, prompt)
    
    def normal_qa(self, metadata):
        text_dict = self.process_text(metadata["answer"], metadata["question"])
        return text_dict

    def normal_caption(self, metadata):
        if self.args.no_prompt:
            text_dict = self.process_text(metadata["answer"], "")
        else:
            prompt = random.choice(self.templates["Caption"]).replace('<image>', '')
            text_dict = self.process_text(metadata["answer"], prompt)
        return text_dict

    def multichoice(self, metadata):
        def generate_prompt_in_multi_choice(choices, question):
            language_zh = is_chinese(question)
            if language_zh:
                choice_prompt = random.choice(self.templates_zh["Choices"])
            else:
                choice_prompt = random.choice(self.templates_en["Choices"])
            prompt = question + "\n" + choice_prompt + "\n"
            start_op = 'A'
            for item in choices:
                prompt += f'{start_op}: {item}\n'
                start_op = chr(ord(start_op) + 1)
            prompt += "回答: " if language_zh else "Answer: "
            return prompt
        prompt = generate_prompt_in_multi_choice(metadata["choices"], metadata["question"])
        text_dict = self.process_text(metadata["answer"], prompt)
        return text_dict
