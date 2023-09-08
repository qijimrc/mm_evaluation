import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes, save_data

DATASET_NAWE = "Screen2Words"

def process_data(root_dir, mode):
    filename = os.path.join(root_dir, f"raw/screen2words/split/{mode}_screens.txt")
    img_dir = os.path.join(root_dir, f"raw/screen2words/unique_uis/combined")
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAWE}/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(filename, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        lines = set([int(line.strip()) for line in lines])
    all_results = {}
    drop_num, item_num = 0, 0
    df = pd.read_csv(os.path.join(root_dir, 'raw/screen2words/screen_summaries.csv'))
    df = df[df["screenId"].isin(lines)]
    for idx, group_df in tqdm.tqdm(df.groupby(by="screenId")):
        image_path = os.path.join(img_dir, f"{idx}.jpg")
        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
        if image_path not in all_results:
            all_results[image_path] = []
        for i, row in group_df.iterrows():
            c_data = {
                "datatype": "normal_caption",
                "question_id": "%09d" %item_num,
                "metadata": {
                    "answer": row["summary"]
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
    for mode in ['train', 'dev', 'test']:
        print(f"process {mode}.")
        process_data(root_dir, mode)