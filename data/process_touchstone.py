import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAWE = "TouchStone"

def process_data(root_dir, mode):
    filename = os.path.join(root_dir, f"raw/TouchStone/ts_sample.tsv")
    img_dir = os.path.join(root_dir, f"raw/TouchStone/ts_images")
    save_dir = os.path.join(root_dir, f"processed/TouchStone/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = pd.read_csv(filename, sep="\t")
    all_results = {}
    drop_num, item_num = 0, 0
    for i, sub_data in tqdm.tqdm(data.iterrows()):
        image_path = os.path.join(img_dir, os.path.basename(sub_data["image"]))
        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
            continue
        if image_path not in all_results:
            all_results[image_path] = []
        c_data = {
            "datatype": "normal_qa",
            "question_id": sub_data["index"],
            "metadata": {
                "question":sub_data["question"],
                "answer": sub_data["gpt4_ha_answer"],
                "human_annotation": sub_data["human_annotation"],
                "category": sub_data["category"],
                "task_name": sub_data["task_name"]
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
    root_dir = "/share/img_datasets/mmbench_datasets"
    for mode in ['test']:
        print(f"process {mode}.")
        process_data(root_dir, mode)