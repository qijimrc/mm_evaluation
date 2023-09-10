import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAWE = "ChatQA"

def process_data(root_dir, mode):
    filename = os.path.join(root_dir, f"raw/ChatQA/Dataset/{mode}/{mode}_human.json")
    img_dir = os.path.join(root_dir, f"raw/ChatQA/Dataset/{mode}/png")
    save_dir = os.path.join(root_dir, f"processed/ChatQA/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    all_results = {}
    drop_num, item_num = 0, 0
    for sub_data in tqdm.tqdm(data):
        image_path = os.path.join(img_dir, sub_data["imgname"])
        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
        if image_path not in all_results:
            all_results[image_path] = []
        c_data = {
            "datatype": "normal_qa",
            "question_id": "%09d" %item_num,
            "metadata": {
                "question":sub_data["query"],
                "answer": sub_data["label"]
            }
        }
        all_results[image_path].append(c_data)
        item_num += 1
    # save tarfiles
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ['train', 'val', 'test']:
        print(f"process {mode}.")
        process_data(root_dir, mode)