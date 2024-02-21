import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter
from utils import get_image_bytes, save_data


# with open('templates.json') as f:
#     templates = json.load(f)
#     cap_template = templates['Caption']

DATASET_NAWE = "NoCaps"

def process_data(root_dir):
    anns_file = os.path.join(root_dir, f"raw/NoCaps/nocaps_val_4500_captions.json")
    img_dir = "raw/NoCaps/imgs"    
    save_dir = os.path.join(root_dir, f"processed/NoCaps-New/val")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_captions = {}
    with open(anns_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
        for ann in data['annotations']:
            if ann["image_id"] not in img_captions:
                img_captions[ann["image_id"]] = []
            img_captions[ann["image_id"]].append(ann["caption"])
        id2imgs = {info['id']: info for info in data['images']}

    all_results = {}
    drop_num, item_num = 0, 0
    for image_id, caption_list in tqdm.tqdm(img_captions.items()):
        imginfo = id2imgs[image_id]
        image_path = os.path.join(root_dir, img_dir, imginfo['file_name'])
        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
        if image_path not in all_results:
            all_results[image_path] = []
        c_data = {
            "datatype": "normal_caption",
            "question_id": "%09d" %item_num,
            "metadata": {
                "answer": caption_list[0],
                "answer_list": caption_list,
                'open_image_id': imginfo['open_images_id'],
                'domain': imginfo['domain'],
                "image_id": ann["image_id"]
            }
        }
        all_results[image_path].append(c_data)
        item_num += 1
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, "test")
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/share/img_datasets/mmbench_datasets"
    process_data(root_dir)
