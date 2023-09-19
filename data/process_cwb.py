import os
import json
import random

from utils import save_data

DATASET_NAWE = "CWB"

import re

def fix_spacing(text):
    # 去除标点符号前的空格
    text = re.sub(r' \.', '.', text)
    text = re.sub(r' ,', ',', text)
    text = re.sub(r' \?', '?', text)
    text = re.sub(r' !', '!', text)
    text = re.sub(r' ;', ';', text)
    text = re.sub(r' :', ':', text)
    text = re.sub(r' \$', ':', text)

    # 循环处理引号，直到没有要处理的引号为止
    while re.search(r' " ([^"]+?) "', text):
        text = re.sub(r' " ([^"]+?) "', r' "\1"', text)
    
    text = re.sub(r' " (.*?)"$', r' "\1"', text)

    return text

def process_data(root_dir, mode):
    img_dir = os.path.join(root_dir, f"raw/CWB/flickr30k-images")
    suffix = 'eval' if mode == 'val' else mode
    filename = os.path.join(root_dir, f"raw/CWB/CWB_flickr30k_{suffix}.jsonl")
    save_dir = os.path.join(root_dir, f"processed/CWB/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_results = {}
    drop_num, item_num = 0, 0
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp:
            c_data = json.loads(line)
            image_path = os.path.join(img_dir, c_data["image_id"]+".jpg")
            if not os.path.exists(os.path.join(root_dir, image_path)):
                print(f"not found: {image_path}")
                drop_num += 1
                continue
            if image_path not in all_results:
                all_results[image_path] = []
            c_data = {
                "datatype": "grounding_caption",
                "question_id": c_data["id"],
                "metadata": {
                    "caption":fix_spacing(c_data["sentence"]),
                    "caption_boxes": c_data["boxes_seq"],
                    "boxes": c_data["boxes"]
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
    root_dir = "/mnt/shared/img_datasets/grounding_stage2"
    for mode in ['train', 'val', 'test']:
        print(f"process {mode}.")
        process_data(root_dir, mode)