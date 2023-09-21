import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAME = "STVQA"

def process_data(root_dir, mode):
    filename = os.path.join(root_dir, f"raw/ST-VQA")
    img_dir = os.path.join(root_dir, "raw/ST-VQA")
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAME}/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    question_ids = set()
    if mode == "train":
        data = []
        for path_id in [3]:
            with open(os.path.join(filename, f"{mode}_task_{path_id}.json"), "r", encoding="utf-8") as fp:
                data.extend(json.load(fp)['data'])
    else:
        with open(os.path.join(filename, f"{mode}.json"), "r", encoding="utf-8") as fp:
            data = json.load(fp)['data']
    all_results = {}
    drop_num, item_num = 0, 0
    for c_data in tqdm.tqdm(data):
        if c_data["set_name"] not in mode:
            continue
        if 'answers' not in c_data:
            answer = ''
            answer_list = []
        else:
            counts = Counter(c_data["answers"])
            answer = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
            answer_list = c_data["answers"]
        image_path = os.path.join(img_dir, c_data["file_path"])
        if not os.path.exists(os.path.join(root_dir, image_path)):
            print(f"not found: {image_path}")
            drop_num += 1
            continue
        if image_path not in all_results:
            all_results[image_path] = []
        c_data = {
            "datatype": "normal_qa",
            "question_id": c_data["question_id"],
            "metadata": {
                "question": c_data["question"],
                "answer": answer,
                "answer_list": answer_list
            }
        }
        if c_data["question_id"] in question_ids:
            print(f"[{mode}]: find repeated question_ids, {c_data['question_id']}")
        else:
            question_ids.add(c_data["question_id"])
            all_results[image_path].append(c_data)
            item_num += 1
    # save tarfiles
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAME, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/mnt/shared/img_datasets/mmbench_datasets"
    for mode in ['train', 'test_task_1','test_task_2','test_task_3']:
        print(f"process {mode}.")
        process_data(root_dir, mode)