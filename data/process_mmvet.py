import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAME = "MM-Vet"

def process_data(root_dir):
    filename = os.path.join(root_dir, f"raw/MM-Vet/mm-vet.json")
    img_dir = os.path.join(root_dir, "raw/MM-Vet/images")
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAME}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    question_ids = set()
    with open(filename, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    all_results = {}
    drop_num, item_num = 0, 0
    for imkey in tqdm.tqdm(data):
        c_data = data[imkey]
        image_path = os.path.join(img_dir, c_data["imagename"])
        if not os.path.exists(os.path.join(root_dir, image_path)):
            print(f"not found: {image_path}")
            drop_num += 1
            continue
        if image_path not in all_results:
            all_results[image_path] = []
        c_data = {
            "datatype": "normal_qa",
            "question_id": imkey,
            "metadata": {
                "question": c_data["question"],
                "answer": c_data["answer"],
                "capability": c_data["capability"],
                "imagesource": c_data["imagesource"]
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