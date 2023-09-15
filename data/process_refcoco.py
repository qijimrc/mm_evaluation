# ref: shikra
import os
import math
import json
import tqdm
import random
import pickle
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAME = "RefCOCO"

test_val_image = set()

def process_data(root_dir, mode):
    dir_name = mode if mode != "val" else "refcoco-val"
    img_dir = os.path.join(root_dir, f"/nxchinamobile2/shared/img_datasets/MSCOCO/MSCOCO2014/train2014/")
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAME}/{dir_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_results = {}
    data_types = {}
    drop_num, item_num, train_drop_num = 0, 0, 0
    for dataset in ["refcoco", "refcoco+", "refcocog"]:
        _tmp = "umd" if dataset == "refcocog" else "unc"
        filename = os.path.join(root_dir, f"raw/RefCOCO/{dataset}/instances.json")
        ans_file = os.path.join(root_dir, f"raw/RefCOCO/{dataset}/refs({_tmp}).p")
        with open(filename, "r", encoding="utf-8") as fp:
            instances = json.load(fp)['annotations']
        
        id2insts =  {inst['id']: inst for inst in instances} # this id aligns with the `ann_id` in res.p
        with open(ans_file, 'rb') as f:
            annotations = pickle.load(f)

        for ann in tqdm.tqdm(annotations):
            if mode == "train" and ann['split'] != "train":
                continue
            if mode == "val" and (ann["split"] != "val" or dataset != "refcoco"):
                continue
            if mode == "test" and ann['split'] == "train":
                continue
            c_type = f'{dataset}-{ann["split"]}'
            if c_type not in data_types:
                data_types[c_type] = 0
            image_path = os.path.join(img_dir, '_'.join(ann['file_name'].split("_")[:-1]) + ".jpg")
            if not os.path.exists(image_path):
                print(f"not found: {image_path}")
                drop_num += 1
                continue
            if mode == "test" or mode == "val":
                test_val_image.add(image_path)
            else:
                if image_path in test_val_image:
                    train_drop_num += 1
                    continue

            if image_path not in all_results:
                all_results[image_path] = []
            inst = id2insts[(ann['ann_id'])]
            bbox = inst['bbox']
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2],  bbox[1]+bbox[3]]
            for sent in ann['sentences']:
                data_types[c_type] += 1
                c_data = {
                    "datatype": "grounding_qa",
                    "question_id": "%09d" %item_num,
                    "metadata": {
                        "question":sent["sent"],
                        "answer": "<ph_st><ph_ed>",
                        "question_boxes": [],
                        "answer_boxes": [[0]],
                        "boxes": [bbox],
                        "type": c_type
                    }
                }
                all_results[image_path].append(c_data)
                item_num += 1
    print(f"Find types: {data_types}")
    # save tarfiles
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAME, mode)
    print(f"Save: {image_num} images, {item_num} samples. Image Drop: {drop_num} samples. Train Drop: {train_drop_num} samples.")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ['val', 'test', 'train']:
        print(f"process {mode}.")
        process_data(root_dir, mode)