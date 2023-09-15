import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAWE = "VizWiz-VQA"

def process_data(root_dir, mode):
    filename = os.path.join(root_dir, f"raw/VizWiz-VQA/vqa_annotation/{mode}.json")
    img_dir = os.path.join(root_dir, f"raw/VizWiz-VQA/{mode}")
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAWE}/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(filename, 'r') as fp:
        data = json.load(fp)
    all_results = {}
    drop_num, item_num = 0, 0
    for c_data in tqdm.tqdm(data):
        if 'answers' not in c_data:
            answer = ''
            answer_list = []
        else:
            answer_list = [s["answer"] for s in c_data["answers"]]
            counts = Counter(answer_list)
            answer = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
        image_path = os.path.join(img_dir, c_data["image"])
        if not os.path.exists(os.path.join(root_dir, image_path)):
            print(f"not found: {image_path}")
            drop_num += 1
            continue
        if image_path not in all_results:
            all_results[image_path] = []
        if mode == "test":
            c_data = {
                "datatype": "normal_qa",
                "question_id": "%09d" %item_num,
                "metadata": {
                    "question": c_data["question"],
                    "answer": "",
                    "answer_list": [],
                    "answer_type": "",
                    "answerable": -1
                }
            }
        else:
            c_data = {
                "datatype": "normal_qa",
                "question_id": "%09d" %item_num,
                "metadata": {
                    "question": c_data["question"],
                    "answer": answer,
                    "answer_list": answer_list,
                    "answer_type": c_data["answer_type"],
                    "answerable": c_data["answerable"]
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
    for mode in ['test']:
        print(f"process {mode}.")
        process_data(root_dir, mode)