import io
import os
import jsonlines
import pandas as pd
import webdataset as wds

from PIL import Image
from tqdm import tqdm
from utils import get_image_bytes, generate_prompt_in_multi_choice

def process_data(raw_dir, save_dir, img_dir, mode):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    all_data = {}
    drop_num, result_tar, tar_id, item_num = 0, [], 0, 0
    eval_type_dict = {"0001": "existence", "0002": "color", "0003": "position"} 
    with jsonlines.open(os.path.join(raw_dir, f"raw/HalVQA/data_{mode}.jsonl"), "r") as fp:
        for data in fp:
            image_path = os.path.join(img_dir, data["image"])
            if not os.path.exists(image_path):
                print(f'not found: {image_path}')
                drop_num += 1
                continue
            eval_type = eval_type_dict[data["eval_type"]] if data["eval_type"] in eval_type_dict.keys() else data["eval_type"]
            json_data = {
                "question_id": "%06d" %item_num,
                "eval_type": eval_type,
                "answer": data["answer"],
                "prompt": generate_prompt_in_multi_choice(data["choices"], data["question"], language="zh"),
            }
            if image_path not in all_data:
                all_data[image_path] = []
            all_data[image_path].append(json_data)
            item_num += 1

    image_num = 0
    for image_path, meta_data in tqdm(all_data.items()):
        c_tar = {
            "__key__": "%06d" %image_num,
            "jpg": get_image_bytes(image_path),
            "json": meta_data
        }
        result_tar.append(c_tar)
        image_num += 1
        if len(result_tar) >= 1000:
            with wds.TarWriter(os.path.join(save_dir, f"{mode}_halvqa_%06d.tar" %(tar_id)), "w") as tar:
                for res in result_tar:
                    tar.write(res)
            result_tar = []
            tar_id += 1
    if len(result_tar) > 0:
        with wds.TarWriter(os.path.join(save_dir, f"{mode}_halvqa_%06d.tar" %(tar_id)), "w") as tar:
            for res in result_tar:
                tar.write(res)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    img_dir = os.path.join(root_dir, "raw/HalVQA/images")
    for mode in ["train", "test"]:
        print(f'process {mode}')
        save_dir = os.path.join(root_dir, f"processed/HalVQA/{mode}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        process_data(root_dir, save_dir, img_dir, mode)