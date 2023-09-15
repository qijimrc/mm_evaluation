import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAWE = "DocVQA"

def process_data(root_dir, mode):
    img_dir = os.path.join(root_dir, f"raw/DocVQA/Task1_Single_Page_Document_Visual_Question_Answering/{mode}")
    filename = os.path.join(root_dir, f"raw/DocVQA/Task1_Single_Page_Document_Visual_Question_Answering/{mode}/{mode}_v1.0.json")
    save_dir = os.path.join(root_dir, f"processed/DocVQA/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)['data']
    all_results = {}
    drop_num, item_num = 0, 0
    for c_data in tqdm.tqdm(data):
        if c_data["data_split"] != mode:
            continue
        if mode == "test":
            answer = ""
            answer_list = []
        else:
            counts = Counter(c_data["answers"])
            answer = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
            answer_list = c_data["answers"]
        image_path = os.path.join(img_dir, c_data["image"])
        if not os.path.exists(os.path.join(root_dir, image_path)):
            print(f"not found: {image_path}")
            drop_num += 1
            continue
        if image_path not in all_results:
            all_results[image_path] = []
        c_data = {
            "datatype": "normal_qa",
            "question_id": c_data["questionId"],
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
    for mode in ['train', 'val', 'test']:
        print(f"process {mode}.")
        process_data(root_dir, mode)