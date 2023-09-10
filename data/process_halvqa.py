import io
import os
import random
import jsonlines
import pandas as pd
import webdataset as wds

from PIL import Image
from tqdm import tqdm
from utils import get_image_bytes, save_data

DATASET_NAWE = "HalVQA"

def process_data(root_dir, save_dir, img_dir, mode):
    all_data = {}
    drop_num, item_num = 0, 0
    eval_type_dict = {"0001": "existence", "0002": "color", "0003": "position"} 
    with jsonlines.open(os.path.join(root_dir, f"raw/HalVQA/data_{mode}.jsonl"), "r") as fp:
        for data in fp:
            image_path = os.path.join(img_dir, data["image"])
            if not os.path.exists(image_path):
                print(f'not found: {image_path}')
                drop_num += 1
                continue
            eval_type = eval_type_dict[data["eval_type"]] if data["eval_type"] in eval_type_dict.keys() else data["eval_type"]
            try:
                json_data = {
                    "datatype": "multichoice",
                    "question_id": "%06d" %item_num,
                    "metadata": {
                        "question": data["question"],
                        "choices": data["choices"],
                        "answer": data["choices"].index(data["answer"]),
                        "type": eval_type,
                    }
                }
            except Exception as e:
                print(e)
                drop_num += 1
                continue
            if image_path not in all_data:
                all_data[image_path] = []
            all_data[image_path].append(json_data)
            item_num += 1
    all_data = [{"image_path": key, "json": value} for key, value in all_data.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    img_dir = os.path.join(root_dir, "raw/HalVQA/images")
    for mode in ["train", "test"]:
        print(f'process {mode}')
        save_dir = os.path.join(root_dir, f"processed/HalVQA/{mode}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        process_data(root_dir, save_dir, img_dir, mode)
