import io
import os
import random
import jsonlines
import pandas as pd
import webdataset as wds

from PIL import Image
from tqdm import tqdm
from utils import save_data

from pdb import set_trace as st
import jsonlines

from torch.utils.data import Dataset
import base64, io
import json

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 save_dir,
                 sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt
        self.save_dir = save_dir

    def __len__(self):
        return len(self.df)

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image_enc = self.df.iloc[idx]['image']
        try:
            image = decode_base64_to_image(image_enc)
        except:
            print(f"Error decoding image {index}")
            st()
            return None
            # return None
        
        image_idx = "%09d" %index
        image_save_path = os.path.join(self.save_dir, f"{image_idx}.png")
        if not os.path.exists(image_save_path):
            image.save(image_save_path)
        
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else '?'
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = "" #f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        circular_options = []
        circular_answers = []
        answer_text = options[answer] if answer != '?' else '?'
        option_ids = []
        option_texts = []
        for key, item in options.items():
            option_ids.append(key)
            option_texts.append(item)
        for shift in range(1, len(options)):
            new_options = ""
            new_answer = ""
            for i in range(len(options)):
                key = option_ids[i]
                item = option_texts[(i + shift) % len(options)]
                if item == answer_text:
                    new_answer = key
                new_options += f'{key}. {item}\n'
            if answer_text == "?":
                new_answer = "?"
            assert new_options != ""
            assert new_answer != ""
            circular_options.append(new_options)
            circular_answers.append(new_answer)

        hint = self.load_from_df(idx, 'hint')
        data = {
            'img_path': image_save_path,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_dict': options,
            'index': str(index),
            'context': hint,
            'circular_options': circular_options,
            'circular_answers': circular_answers,
        }
        return data

def build_prompt(context, question, options, split):
    
    if context is not None:
        prompt = context + '\n\n' + question + '\n' + options
    else:
        prompt = question + '\n' + options

    if "_CN" not in split:
        prompt = prompt + "\n" + "Short answer."
    else:
        prompt = prompt + "\n" + "Short answer."
        
    # if "_CN" not in split:
    #     prompt = prompt + "\n" + "Answer with the option's letter from the given choices directly."
    # else:
    #     prompt = prompt + "\n" + "请直接回答选项字母。"
    assert prompt is not None
    return prompt

DATASET_NAME = "MMMU"
def process_data(root_dir, split):
    # filename = os.path.join(root_dir, f"meta.jsonl")
    # img_dir = os.path.join(root_dir, "images")
    save_dir = root_dir.replace("datasets", "datasets_processed")
    save_dir = os.path.join(save_dir, split)
    save_images_dir = os.path.join(save_dir, "images")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_images_dir, exist_ok=True)
    
    data_file = os.path.join(root_dir, f"{split}_legacy.tsv")
    print(f"Save to {save_dir}")
    # run for each subject
    dataset = MMBenchDataset(data_file=data_file, save_dir=save_images_dir)
    
    # read using jsonlines
    all_results = {}
    drop_num, item_num = 0, 0   
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        
        image_idx = "%09d" %int(sample['index'])
        
        image_path = sample["img_path"]
        if image_path not in all_results:
            all_results[image_path] = []
        
        # base QA
        query = build_prompt(context=sample['context'], 
                             question=sample['question'], 
                             options=sample['options'], 
                             split=split)

        circ_id = 0
        circ_idx = "%09d"%circ_id
        question_id = f"{image_idx}_{circ_idx}"

        c_data = {
            "datatype": "normal_qa",
            "question_id": question_id,
            "metadata": {
                "question": query,
                "answer": sample["answer"] if sample["answer"] is not None else "?",
                'circ_idx': circ_idx,
                "orig_dict": sample.copy(),
            }
        }
        all_results[image_path].append(c_data)
        # circular QA
        if len(sample['circular_options']) > 0:
            for c_idx in range(len(sample['circular_options'])):
                query = build_prompt(context=sample['context'], 
                                     question=sample['question'], 
                                     options=sample['circular_options'][c_idx], 
                                     split=split)
                circ_id = c_idx + 1
                circ_idx = "%09d"%circ_id
                question_id = f"{image_idx}_{circ_idx}"
                circ_ans = sample["circular_answers"][c_idx]
                c_data = {
                    "datatype": "normal_qa",
                    "question_id": question_id,
                    "metadata": {
                        "question": query,
                        "answer": circ_ans if circ_ans is not None else "?",
                        'circ_idx': circ_idx,
                        "orig_dict": sample.copy(),
                    }
                }
                all_results[image_path].append(c_data)
        # st()
        # # print json in a pretty way
        # print(json.dumps(c_data, indent=4))
        # all_results[image_path].append(c_data)
        item_num += 1
        
    
    
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, split, "")
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")
    
    


if __name__ == "__main__":
    root_dir = "/share/home/chengyean/evaluation/cya_ws/datasets/MMBench"
    
    mode_file_ls = [
        # "MMBench_DEV_CN", 
        "MMBench_TEST_CN", "MMBench_DEV_EN", "MMBench_TEST_EN"
    ]
    
    for mode in mode_file_ls:
        print(f'process {mode}')
        process_data(root_dir, mode)
