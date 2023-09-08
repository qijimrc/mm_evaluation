import os
import json
import glob
import jsonlines
import random
from tqdm import tqdm
import pandas as pd
import webdataset as wds

from utils import get_image_bytes, save_data

DATASET_NAWE = "GQA"

def process_data(filename_list, save_dir, root_dir, img_dir):
    # 汇总数据
    im2data = {}
    drop_num = 0
    item_num = 0
    for filename in filename_list:
        with open(filename, "r", encoding='utf-8') as f:
            processed_datas = list(json.load(f).values())
        print(f"process {len(processed_datas)} samples in {filename}")
        for i, data in tqdm(enumerate(processed_datas)):
            if 'imageId' not in data or 'question' not in data or 'answer' not in data  or 'fullAnswer' not in data:
                drop_num += 1
                continue
            image_path = os.path.join(root_dir, img_dir, data['imageId'] + ".jpg")
            if not os.path.exists(image_path):
                print(f'not found: {image_path}')
                drop_num += 1
                continue
            prompt, txt, full_answer = data['question'], data['answer'], data['fullAnswer']
            conversation = {
                "datatype": "normal_qa",
                "question_id": "%09d" %(item_num),
                "metadata": {
                    "question":prompt,
                    "answer": txt,
                    "full_answer": full_answer
                }
            }
            item_num += 1
            if image_path not in im2data:
                im2data[image_path] = []
            im2data[image_path].append(conversation)
    all_data = [{"image_path": key, "json": value} for key, value in im2data.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")


if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    img_dir = "raw/GQA/images"
    for mode in ["train", "val", "test"]:
        print(f'process {mode}')
        save_dir = os.path.join(root_dir, f"processed/{DATASET_NAWE}/{mode}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # merge image data
        if mode == "train":
            file_list = glob.glob(os.path.join(root_dir, f"raw/GQA/train_all_questions/*.json"))
        elif mode == "val":
            file_list = [os.path.join(root_dir, f"raw/GQA/val_all_questions.json")]
        else:
            file_list = [os.path.join(root_dir, f"raw/GQA/testdev_all_questions.json")]
        process_data(file_list, save_dir, root_dir, img_dir)
