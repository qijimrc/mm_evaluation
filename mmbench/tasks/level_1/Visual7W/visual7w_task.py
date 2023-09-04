import os
import random
import math
import pandas as pd
import json

from PIL import Image
from typing import Dict, List
from torch.utils.data import Dataset
from sat.helpers import print_rank0

from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

def refine_box(box, scale, new_width, new_height):
    box = [min(round(box[0]*scale), new_width-1), min(round(box[1]*scale), new_height-1), min(round(box[2]*scale), new_width-1), min(round(box[3]*scale), new_height-1)]
    box = [box[0]/new_width, box[1]/new_height, box[2]/new_width, box[3]/new_height]
    box = [math.floor(x*1000) for x in box]
    if box[0] >= 1000 or box[1] >= 1000 or box[2] >= 1000 or box[3] >= 1000:
        print_rank0(str(box))
        box = [min(box[0], 999), min(box[1], 999), min(box[2], 999), min(box[3], 999)]
    return box

def get_text_by_box(boxlist, sep=" "):
    strs = [f"{box[0]:03d},{box[1]:03d},{box[2]:03d},{box[3]:03d}" for box in boxlist]
    random.shuffle(strs)
    return "{}[[{}]]".format(sep, ";".join(strs))

def parse_resize(img, h, w):
    # if type(img_bytes) is not Image.Image:
    #     img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # else:
    #     img = img_bytes.convert('RGB')
    totalpatch, lpatch = h, w
    # maximize scale s.t.
    scale = math.sqrt(totalpatch * (lpatch / img.size[1]) * (lpatch / img.size[0]))
    num_feasible_rows = max(min(math.floor(scale * img.size[1] / lpatch), totalpatch), 1)
    num_feasible_cols = max(min(math.floor(scale * img.size[0] / lpatch), totalpatch), 1)
    target_height = max(num_feasible_rows * lpatch, 1)
    target_width = max(num_feasible_cols * lpatch, 1)
    # img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
    # img = img.crop((0, 0, target_width, target_height))
    return scale, target_width, target_height

def get_box(b):
    return max(b['x'], 0), max(b['y'], 0), b['x']+b['width'], b['y']+b['height']

def build_sample_input(question, answer, boxes, scale, new_width, new_height):
    new_boxes = [refine_box(box, scale, new_width, new_height) for box in boxes]
    box_txt = [get_text_by_box([box], sep="") for box in new_boxes]
    prompt = f"""{question} Select from:\nA. {box_txt[0]}\nB. {box_txt[1]}\nC. {box_txt[2]}\nD. {box_txt[3]}\n Answer: """
    txt = chr(ord('A')+answer)
    return prompt, txt

class Visual7wDataset(Dataset):
    def __init__(self, path, args, mt):
        super().__init__()
        data_file, split = path.split('---')
        with open(data_file) as f:
            data = json.load(f)
        box = data['boxes']
        key2box = {}
        for b in box:
            key2box[b['box_id']] = get_box(b)
        images = [d for d in data['images'] if d['split'] == split]
        self.key2box = key2box
        self.samples = []
        for d in images:
            self.samples.extend([{**dic, 'filename':d['filename']} for dic in d['qa_pairs']])

        self.mt = mt
        self.data_home_dir = args.data_home_dir

    def __getitem__(self, index):
        sample = self.samples[index]
        img_path = os.path.join(self.data_home_dir, "Visual7W/raw/images", sample['filename'])
        img = Image.open(img_path).convert('RGB')
        boxes = []
        boxes.append(self.key2box[sample['answer']])
        for q in sample['multiple_choices']:
            boxes.append(self.key2box[q])
        permute = [0, 1, 2, 3]
        random.shuffle(permute)
        rand_boxes = [boxes[i] for i in permute]
        answer = permute.index(0)

        img_dict = {'vision': self.mt.image_processor(img)}
        if self.mt.cross_image_processor:
            img_dict.update({'cross': self.mt.cross_image_processor(img)})
        
        scale, new_width, new_height = parse_resize(img, 400, 14)
        prompt, txt = build_sample_input(sample['question'], answer, rand_boxes, scale, new_width, new_height)
        prompt = self.mt.text_processor.history_to_prompt([], prompt, add_eoi_first=True)
        text_dict = self.mt.text_processor(txt, prompt)
        if text_dict is None:
            assert "You should not run to here!"
        ret = {'question_id': sample['qa_id'], 'label_text': txt}
        ret.update(text_dict)
        ret.update(img_dict)
        return ret
    
    def __len__(self):
        return len(self.samples)

@Registry.register_task('Visual7W')
class Visual7W(BaseTask):
    def __init__(self, task_cfg, custom_functions, **kw_args):
        self.task_name = 'Visual7W'
        super().__init__(task_cfg, custom_functions, **kw_args)
    
    def create_dataset_function(self, mt, path, args):
        dataset = Visual7wDataset(path, args, mt)
        return dataset

    def calc_scores(self, args, results_total, metrics: List[str]=['acc']) -> Dict:
        question_ids, preds, labels = results_total["question_ids"], results_total["preds"], results_total["labels"]
        res_df = pd.DataFrame({"question_ids": question_ids, "preds": preds, "labels": labels})
        # remove duplicates
        res_df = res_df.drop_duplicates(subset=["question_ids"])
        # compute score
        metric_cls = Registry.get_metric_class('acc')
        return {"acc": metric_cls.calc_score(res_df["labels"], res_df["preds"])}
