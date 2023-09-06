import os
import json
import tqdm
import pandas as pd
import webdataset as wds
from collections import Counter

from utils import get_image_bytes


def process_data(root_dir, mode):
    filename = os.path.join(root_dir, f"raw/TextVQA/train_val/TextVQA_0.5.1_{mode}.json")
    img_dir = os.path.join(root_dir, "raw/TextVQA/train_val/train_images")
    save_dir = os.path.join(root_dir, f"processed/TextVQA/{mode}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)['data']
    all_results = {}
    drop_num, item_num = 0, 0
    for c_data in tqdm.tqdm(data):
        counts = Counter(c_data["answers"])
        answer = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
        image_path = os.path.join(img_dir, c_data["image_id"] + ".jpg")
        if not os.path.exists(os.path.join(root_dir, image_path)):
            print(f"not found: {image_path}")
            drop_num += 1
        if image_path not in all_results:
            all_results[image_path] = []
        all_results[image_path].append({
            "question_id": c_data["question_id"],
            "question": c_data["question"],
            "answer": answer
        })
        item_num += 1
    # save tarfiles
    tar_id, result_tar, image_num = 0, [], 0
    for key, value in tqdm.tqdm(all_results.items()):
        c_tar = {
            "__key__": "%06d" %image_num,
            "json": value,
            "jpg": get_image_bytes(key)
        }
        result_tar.append(c_tar)
        image_num += 1
        if len(result_tar) >= 1000:
            with wds.TarWriter(os.path.join(save_dir, f"{mode}_textvqa_%06d.tar" %(tar_id)), "w") as tar:
                for res in result_tar:
                    tar.write(res)
            result_tar = []
            tar_id += 1
    if len(result_tar) > 0:
        with wds.TarWriter(os.path.join(save_dir, f"{mode}_textvqa_%06d.tar" %(tar_id)), "w") as tar:
            for res in result_tar:
                tar.write(res)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ['train', 'val']:
        print(f"process {mode}.")
        process_data(root_dir, mode)