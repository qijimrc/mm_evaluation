import io
import os
import json
import jsonlines
import webdataset as wds

from PIL import Image
from tqdm import tqdm
from utils import get_image_bytes, generate_prompt_in_multi_choice

def process_data(raw_dir, save_dir, img_dir, mode):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    drop_num, result_tar, tar_id, item_num = 0, [], 0, 0
    all_data = []
    with jsonlines.open(os.path.join(raw_dir, f"data_{mode}.jsonl"), "r") as fp:
        for data in fp:
            all_data.append(data)
    for data in tqdm(all_data):
        image_path = os.path.join(img_dir, data["image"])
        c_tar = {
            "__key__": "%06d" %item_num,
            "prompt": generate_prompt_in_multi_choice(data["choices"], data["question"], language="zh"),
            "jpg": get_image_bytes(image_path),
            "answer": data["answer"],
            "ttype": "IMG",
            "etype": data["eval_type"]
        }
        result_tar.append(c_tar)
        item_num += 1
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
    print(f"Save: {item_num}. Drop: {drop_num}")

if __name__ == "__main__":
    raw_dir = "/nxchinamobile2/shared/mmbench_datasets/HalVQA/raw"
    save_dir = "/nxchinamobile2/shared/mmbench_datasets/HalVQA/web_dataset"
    img_dir = os.path.join(raw_dir, "images")
    for mode in ["train", "test"]:
        tmp_save_dir = os.path.join(save_dir, mode)
        process_data(raw_dir, tmp_save_dir, img_dir, mode)