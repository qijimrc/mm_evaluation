import os
import logging
import random
import logging
import jsonlines
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

from mmdoctor.dataset.base_dataset import BaseDataset
from mmdoctor.common.utils import find_all_files
from mmdoctor.common.logger import log


class ItemDataset(Dataset, BaseDataset):
    def __init__(self, args, data_dir, **kwargs):
        super().__init__(args, **kwargs)
        self.data = self.load_data(data_dir)
        self.data_indices = self.split_data(args)
    
    def load_data(self, data_dir):
        all_jsonlines = find_all_files(data_dir, suffix=".jsonl")
        data, qa_num, image_num = [], 0, 0
        for file in all_jsonlines:
            jsonl_dir = os.path.dirname(file)
            with jsonlines.open(file, "r") as reader:
                for json_data in reader:
                    qa_num += len(json_data["json"])
                    image_num += 1
                    if "image_path" in json_data:
                        json_data["image_path"] = os.path.join(jsonl_dir, json_data["image_path"])
                    else:
                        json_data["image_path"] = "<null>"
                    # inference: val / test
                    for qa in json_data["json"]:
                        data.append({"image_path": json_data["image_path"], "json": qa})
        log(f"Find {image_num} image-level samples in {qa_num} qa-level samples in all...")
        # DEBUG-CODE-START
        # These codes are for debugging specific data in s_qids
        # s_qids = set()
        # new_data = []
        # for c_data in data:
        #     if c_data['json']['question_id'] in s_qids:
        #         new_data.append(c_data)
        # data = new_data
        # DEBUG-CODE-END
        if self.args.use_debug_mode:
            return data[:self.args.use_debug_mode]
        return data

    def split_data(self, args):
        rank = args.rank
        world_size = args.world_size
        mp_size = args.model_parallel_size
        rank_group_size = world_size // mp_size
        # add padding data
        if len(self.data) % rank_group_size == 0:
            pad_data_len = 0
        else:
            pad_data_len = (len(self.data) // rank_group_size + 1) * rank_group_size - len(self.data)
        # split data
        distributed_data_indices = []
        for i in range(len(self.data) + pad_data_len):
            if i % rank_group_size == rank // mp_size:
                distributed_data_indices.append(i % len(self.data))
        return distributed_data_indices
    
    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        data = self.data[self.data_indices[index]]
        # img
        if 'image_path' in data and not data["image_path"].startswith("<null>"):
            img_path = data["image_path"]
        elif self.args.pad_noimg:
            img_path = os.path.join(os.path.dirname(__file__), "assets/no_img.png")
        else:
            img_path = None
        # text
        dialogues = data['json']
        uni_key = f'{data["image_path"]}-{dialogues["question_id"]}'
        if dialogues["datatype"] in self.custom_data_hooks:
            processor_func = self.custom_data_hooks[dialogues["datatype"]]
        else:
            processor_func = eval(f'self.{dialogues["datatype"]}')
        question, history = processor_func(
            dialogues["metadata"], uni_key, img_path=img_path)
        # other attr
        ret = {
            "question_id": str(dialogues["question_id"]),
            "image_path": img_path,
            "question": question,
            "history": history,
        }
        return ret
