import os
import json
import tqdm
import torch
import random
import pandas as pd
import webdataset as wds
from typing import List
from itertools import combinations
from collections import Counter
from transformers import AutoModel, AutoTokenizer

from utils import get_image_bytes, save_data

DATASET_NAWE = "VqaV2"

def select_answer_by_confidence(answers):
    confidenced_answers = [answer["answer"] for answer in answers]
    if len(confidenced_answers) == 0:
        return None
    counts = Counter(confidenced_answers)
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return counts[0][0]

def process_data(root_dir, mode):
    save_dir = os.path.join(root_dir, f"processed/VQAV2/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if mode == "test":
        question_file = os.path.join(root_dir, f'raw/VQAV2/v2_OpenEnded_mscoco_{mode}2015_questions.json')
        img_dir = os.path.join(root_dir, "raw/VQAV2/test2015")
        with open(question_file) as f1:
            questions = json.load(f1)
        sorted_qs = sorted(questions['questions'], key=lambda x:x['question_id'])
        sorted_anns = None
    else:
        question_file = os.path.join(root_dir, f'raw/VQAV2/v2_OpenEnded_mscoco_{mode}2014_questions.json')
        annotaion_file = os.path.join(root_dir, f'raw/VQAV2/v2_mscoco_{mode}2014_annotations.json')
        img_dir = os.path.join(root_dir, f"raw/TDIUC/MSCOCO2014_{mode}2014")
        with open(question_file) as f1, open(annotaion_file) as f2:
            questions, anns = json.load(f1), json.load(f2)
        sorted_qs = sorted(questions['questions'], key=lambda x:x['question_id'])
        sorted_anns = sorted(anns['annotations'], key=lambda x:x['question_id'])

    all_data, drop_num, item_num = {}, 0, 0
    for idx in tqdm.tqdm(range(len(sorted_qs))):
        if sorted_anns is not None:
            q_info, a_info = sorted_qs[idx], sorted_anns[idx]
            assert q_info["image_id"] == a_info["image_id"]
            qa_info = q_info | a_info
        else:
            qa_info = sorted_qs[idx]
        if mode == "test":
            img_path = os.path.join(img_dir, 'COCO_test2015_{}{}.jpg'.format(''.join(['0']*(12-len(str(qa_info['image_id'])))), qa_info['image_id']))
        else:
            img_path = os.path.join(img_dir, 'COCO_{}2014_{}{}.jpg'.format(mode, ''.join(['0']*(12-len(str(qa_info['image_id'])))), qa_info['image_id']))
        if not os.path.exists(img_path):
            drop_num += 1
            print(f'not found {img_path}.')
            continue
        # answer = select_answer_by_confidence(qa_info["answers"])
        if mode == "test":
            answer = ""
            question_type = ""
            answer_list = []
        else:
            answer = qa_info["multiple_choice_answer"]
            if answer is None:
                drop_num += 1
                print(f'no confidenced answer!')
                continue
            question_type = qa_info["question_type"]
            answer_list = [ans["answer"] for ans in qa_info["answers"]]
        c_data = {
            "datatype": "normal_qa",
            "question_id": qa_info["question_id"],
            "metadata": {
                "question": qa_info["question"],
                "answer": answer,
                "answer_list": answer_list,
                "question_type": question_type
            }
        }
        if img_path not in all_data:
            all_data[img_path] = []
        all_data[img_path].append(c_data)
        item_num += 1
    # save tarfiles
    all_data = [{"image_path": key, "json": value} for key, value in all_data.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")
    
if __name__ == '__main__':
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ["train", "val", "test"]:
        print(f"process {mode}.")
        process_data(root_dir, mode)
