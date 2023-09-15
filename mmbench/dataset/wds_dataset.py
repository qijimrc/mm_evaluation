import logging
import json
import random
import pandas as pd

from io import BytesIO
from PIL import Image
from functools import partial
from typing import Dict, List
from torch.utils.data import Dataset
from sat.helpers import print_rank0
from sat.data_utils.datasets import MetaDistributedWebDataset

from mmbench.dataset.base_dataset import BaseDataset

class WdsDataset(BaseDataset):
    def __init__(self, mt, args, data_mode, other_attr=[], **kwargs):
        super().__init__(mt, args, data_mode, other_attr)
        self.image_qa_cache = {} # {uni_qa_key: c_qaid}

    def process_fn_dataset(self, src):
        for data in src:
            # img
            try:
                if 'jpg' in data:
                    img = Image.open(BytesIO(data['jpg'])).convert('RGB')
                else:
                    img = Image.open(self.img_pad).convert('RGB')
            except Exception as e:
                print_rank0(e, level=logging.WARNING)
                continue
            img_dict = self.process_img(img)
            # json
            dialogues = data['json']
            if self.data_mode == "train":
                if self.args.train_data_load_mode == "random":
                    dialogues = [random.choice(dialogues)]
                elif self.args.train_data_load_mode == "epoch_round":
                    qa_key = data["key"]
                    # if not cache, start from a random index
                    load_id = (self.image_qa_cache.get(qa_key, random.randint(0, len(dialogues)-1)-1) + 1) % len(dialogues)
                    self.image_qa_cache[qa_key] = load_id
                    dialogues = [dialogues[load_id]]
                else:
                    raise ValueError("Unknown train_data_load_mode: {}".format(self.args.train_data_load_mode))
            for qa in dialogues:
                ret = {"question_id": qa["question_id"]}
                text_dict = eval(f'self.{qa["datatype"]}')(qa["metadata"])
                if text_dict == None:
                    print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {qa['metadata']}", level=logging.WARNING)
                    continue
                ret.update(text_dict)
                ret.update(img_dict)
                for attr in self.other_attr:
                    if attr in qa:
                        ret[attr] = qa[attr]
                yield ret
