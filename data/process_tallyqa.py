import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAWE = "TallyQA"

def process_data(root_dir, mode):
    filename = os.path.join(root_dir, f"raw/TallyVQA/{mode}.json")
    img_dir_vg = "/mnt/shared/img_datasets/VG_100K_images"
    img_dir_coco = '/mnt/shared//img_datasets/MSCOCO/MSCOCO2014'
    save_dir = os.path.join(root_dir, f"processed/TallyQA_with_qtype/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    all_results = {}
    drop_num, item_num = 0, 0
    for c_data in tqdm.tqdm(data):
        if 'VG' in c_data['image']:
            img_dir = img_dir_vg
        else:
            img_dir = img_dir_coco

        image_path = os.path.join(img_dir, c_data["image"])
        if not os.path.exists(os.path.join(root_dir, image_path)):
            print(f"not found: {image_path}")
            drop_num += 1
        if image_path not in all_results:
            all_results[image_path] = []
        question_type = None
        if mode == 'test':
            question_type = 'simple' if c_data.get('issimple', None) == True else 'complex'
        c_data = {
            "datatype": "normal_qa",
            "question_id": c_data["question_id"],
            "metadata": {
                "question": c_data["question"],
                "answer": c_data["answer"],
                "question_type": question_type
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
    root_dir = "/mnt/shared/img_datasets/mmbench_datasets"
    for mode in ['test']:
        print(f"process {mode}.")
        process_data(root_dir, mode)