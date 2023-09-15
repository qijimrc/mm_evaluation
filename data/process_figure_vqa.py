import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAWE = "FigureVQA"

def process_data(root_dir, mode):
    if mode == 'train':
        sub_mode = ['train1']
    if mode == 'val':
        sub_mode = ['validation1', 'validation2']
    if mode == 'test':
        sub_mode = ['no_annot_test1', 'no_annot_test2']
    save_dir = os.path.join(root_dir, f"processed/Figure-VQA/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_root_dir = os.path.join(root_dir, 'raw/FigureVQA')
    all_results = {}
    drop_num, item_num = 0, 0
    for m in sub_mode:
        with open(os.path.join(img_root_dir, m, f"qa_pairs.json"), "r", encoding="utf-8") as fp:
            img_dir = os.path.join(img_root_dir, m , 'png')
            data = json.load(fp)['qa_pairs']
            for c_data in tqdm.tqdm(data):
                image_path = os.path.join(img_dir, str(c_data["image_index"]) + '.png')
                if not os.path.exists(image_path):
                    print(f"not found: {image_path}")
                    drop_num += 1
                if image_path not in all_results:
                    all_results[image_path] = []
                if 'answer' not in c_data:
                    answer = ''
                elif int(c_data['answer']) == 1:
                    answer = 'yes'
                else:
                    if c_data['answer'] != 0:
                        print(c_data['answer'])
                        continue
                    answer = 'no'
                c_data = {
                    "datatype": "normal_qa",
                    "question_id": c_data["question_id"],
                    "metadata": {
                        "question": c_data["question_string"],
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
    for mode in ['train','val', 'test']:
        print(f"process {mode}.")
        process_data(root_dir, mode)