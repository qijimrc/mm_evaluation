import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAWE = "ST-VQA"

def process_data(root_dir, mode):
    filename = os.path.join(root_dir, f"raw/ST-VQA")
    img_dir = os.path.join(root_dir, "raw/ST-VQA")
    save_dir = os.path.join(root_dir, f"processed/ST-VQA/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = []
    for path_id in [1,2,3]:
        with open(os.path.join(filename, f"{mode}_task_{path_id}.json"), "r", encoding="utf-8") as fp:
            data.extend(json.load(fp)['data'])
    all_results = {}
    drop_num, item_num = 0, 0
    for c_data in tqdm.tqdm(data):
        if c_data["set_name"] != mode:
            continue
        counts = Counter(c_data["answers"])
        answer = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
        image_path = os.path.join(img_dir, c_data["file_path"])
        if not os.path.exists(os.path.join(root_dir, image_path)):
            print(f"not found: {image_path}")
            drop_num += 1
        if image_path not in all_results:
            all_results[image_path] = []
        c_data = {
            "datatype": "normal_qa",
            "question_id": c_data["question_id"],
            "metadata": {
                "question":c_data["question"],
                "answer": answer
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
    for mode in ['train']:
        print(f"process {mode}.")
        process_data(root_dir, mode)