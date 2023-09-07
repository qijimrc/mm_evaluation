import os
import json
import random
import pandas as pd
from io import BytesIO
from PIL import Image
from typing import Dict, List
from torch.utils.data import Dataset
from sat.helpers import print_rank0

class DataProcessor(object):
    def __init__(self, other_attr=[]):
        self.other_attr = other_attr
        self.img_pad = os.path.join(os.path.dirname(__file__), "no_img.png")

        self.image_qa_cache = {} # {uni_qa_key: c_qaid}
        self.processors = {
            "normal_qa": self.normal_qa,
            "normal_caption": self.normal_caption,
            "multichoice": self.multichoice
        }

    def process_fn_dataset(self, args, mt, src):
        for data in src:
            # img
            try:
                if 'jpg' in data:
                    img = Image.open(BytesIO(data['jpg'])).convert('RGB')
                else:
                    img = Image.open(self.img_pad).convert('RGB')
            except Exception as e:
                print_rank0(e)
                continue
            img_dict = {'vision': mt.image_processor(img)}
            if mt.cross_image_processor:
                img_dict.update({'cross': mt.cross_image_processor(img)})
            # json
            dialogues = json.loads(data['json'].decode("utf-8"))
            if len(dialogues) == 0:
                continue
            if args.data_mode == "train":
                if args.train_data_load_mode == "random":
                    dialogues = [random.choice(dialogues)]
                elif args.train_data_load_mode == "epoch_round":
                    qa_key = f'{data["__url__"]}::{qa["question_id"]}'
                    # if not cache, start from a random index
                    load_id = (self.image_qa_cache.get(qa_key, random.randint(0, len(dialogues)-1)-1) + 1) % len(dialogues)
                    self.image_qa_cache[qa_key] = load_id
                    dialogues = [dialogues[load_id]]
                else:
                    raise ValueError("Unknown train_data_load_mode: {}".format(args.train_data_load_mode))
            for qa in dialogues:
                ret = {"question_id": qa["question_id"]}
                processor_ret = self.processors[qa["datatype"]](qa["metadata"], mt)
                if processor_ret = None:
                    print_rank0(f"")
                    continue
                ret.update(processor_ret)
                ret.update(img_dict)
                yield ret
                
    def normal_qa(self, metadata, mt):
        ret = {}
        
        pass

    def normal_caption(self, metadata, mt):
        pass

    def multichoice(self, metadata, mt):
        pass
