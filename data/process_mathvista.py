import os
import re
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter
from datasets import load_dataset

from utils import get_image_bytes, save_data

DATASET_NAME = "MathVista"

def process_data(root_dir):
    parquet_test1 = os.path.join(root_dir, f"raw/MathVista/data/test-00000-of-00002-6b81bd7f7e2065e6.parquet")
    parquet_test2 = os.path.join(root_dir, f"raw/MathVista/data/test-00001-of-00002-6a611c71596db30f.parquet")
    parquet_testmini = os.path.join(root_dir, f"raw/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet")
    dataset = load_dataset('parquet', data_files={'test1': parquet_test1, 'test2': parquet_test2, 'testmini': parquet_testmini})
    dataset = {'test': [ex for ex in dataset['test1']] + [ex for ex in dataset['test2']], 'testmini': [ex for ex in dataset['testmini']]}

    img_dir = os.path.join(root_dir, "raw/MathVista")
    save_dirs = {'test':os.path.join(root_dir, f"processed/{DATASET_NAME}/test"),
                 'testmini': os.path.join(root_dir, f"processed/{DATASET_NAME}/testmini")}
    for save_dir in save_dirs.values():
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    drop_num, item_num = 0, 0
    for subsetname in dataset:
        all_results = {}
        question_ids = set()
        save_dir = save_dirs[subsetname]
        for ex in tqdm.tqdm(dataset[subsetname], desc=f"processing {subsetname}"):
            image_path = os.path.join(img_dir, ex['image'])
            if not os.path.exists(os.path.join(root_dir, image_path)):
                print(f"not found: {image_path}")
                drop_num += 1
                continue
            if image_path not in all_results:
                all_results[image_path] = []
            if ex['query'].startswith('Hint:'): # Remove hint
                ex['query'] = ex['query'][5:].lstrip()
            c_data = {
                "datatype": "normal_qa", # all using normal qa
                "question_id": ex['pid'],
                "metadata": {
                    "question": ex["query"],
                    "answer": ex["answer"],
                    "choices": ex["choices"],
                    "unit": ex["unit"],
                    "precision": ex["precision"],
                    "original_question": ex["question"], # without prepended instruction
                    "question_type": ex["question_type"],
                    "answer_type": ex["answer_type"],
                    "category": ex["metadata"]['category'],
                    "context": ex["metadata"]['context'],
                    "grade": ex["metadata"]['grade'],
                    "language": ex["metadata"]['language'],
                    "skills": ex["metadata"]['skills'],
                    "source": ex["metadata"]['source'],
                    "task": ex["metadata"]['task'],
                    "split": ex["metadata"]['split'],
                }
            }
            if c_data["question_id"] in question_ids:
                print(f"find repeated question_ids, {c_data['question_id']}")
            else:
                question_ids.add(c_data["question_id"])
                all_results[image_path].append(c_data)
                item_num += 1
        # save tarfiles
        all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
        # random.shuffle(all_data)
        image_num = save_data(all_data, save_dir, DATASET_NAME, mode='test')
        print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/share/img_datasets/mmbench_datasets"
    process_data(root_dir)