import os
import logging
import random
import logging
import jsonlines
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0, print_all

from mmbench.dataset.base_dataset import BaseDataset
from mmbench.common.utils import find_all_files


class ItemDataset(Dataset, BaseDataset):
    def __init__(self, mt, args, data_dir, **kwargs):
        super().__init__(mt, args, **kwargs)
        self.data = self.load_data(data_dir)
        self.image_qa_cache = {} # {uni_qa_key: c_qaid}
    
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
        print_rank0(f"find {image_num} image-level samples in {qa_num} qa-level samples in all...")
        # DEBUG-CODE-START
        # These codes are for debugging specific data in s_qids
        # s_qids = set()
        # new_data = []
        # for c_data in data:
        #     if c_data['json']['question_id'] in s_qids:
        #         new_data.append(c_data)
        # data = new_data
        # DEBUG-CODE-END
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # img
        try:
            if 'image_path' in data and not data["image_path"].startswith("<null>"):
                img = Image.open(data["image_path"]).convert('RGB')
            else:
                img = Image.open(self.img_pad).convert('RGB')
        except Exception as e:
            print_all(e, level=logging.WARNING)
            return {}
        # text
        dialogues = data['json']
        assert len(dialogues) >= 1, f"json length <= 1 in {data}"
        uni_key = f'{data["image_path"]}-{dialogues["question_id"]}'
        text_dict, img = eval(f'self.{dialogues["datatype"]}')(
            dialogues["metadata"], uni_key, img=img)
        img_dict = self.process_img(img)
        if text_dict == None:
            print_all(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {dialogues['metadata']}", level=logging.WARNING)
            return {}
        # other attr
        ret = {**img_dict, **text_dict, "question_id": str(dialogues["question_id"])}
        return ret
