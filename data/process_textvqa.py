import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAWE = "TextVQA"

def process_data(root_dir, mode):
    if mode == "test":
        img_dir = os.path.join(root_dir, "raw/TextVQA/test/test_images")
        filename = os.path.join(root_dir, f"raw/TextVQA/test/TextVQA_0.5.1_test.json")
    else:
        img_dir = os.path.join(root_dir, "raw/TextVQA/train_val/train_images")
        filename = os.path.join(root_dir, f"raw/TextVQA/train_val/TextVQA_0.5.1_{mode}.json")
    save_dir = os.path.join(root_dir, f"processed/TextVQA/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)['data']
    all_results = {}
    drop_num, item_num = 0, 0
    for c_data in tqdm.tqdm(data):
        if mode == "test":
            answer = ""
            answer_list = []
        else:
            counts = Counter(c_data["answers"])
            answer = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
            answer_list = c_data["answers"]
        image_path = os.path.join(img_dir, c_data["image_id"] + ".jpg")
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
                "answer": answer,
                "answer_list": answer_list
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
    # train, val, test
    for mode in ['train', 'val', 'test']:
        print(f"process {mode}.")
        process_data(root_dir, mode)