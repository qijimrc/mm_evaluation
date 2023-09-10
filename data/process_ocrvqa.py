import io
import os
import json
import tqdm
import random
import webdataset as wds

from PIL import Image
from utils import get_image_bytes, save_data

DATASET_NAWE = "OCRVQA"

def process_data(root_dir, save_dir, img_dir, mode):
    drop_num, result_tar, tar_id, item_num = 0, [], 0, 0
    type_dict = {1: "train", 2: "val", 3: "test"}
    all_data = {}
    with open(os.path.join(root_dir, "raw/OCR-VQA/dataset.json"), "r") as fp:
        data = json.load(fp)
        for key, value in tqdm.tqdm(data.items()):
            if type_dict[value["split"]] != mode:
                continue
            image_path = os.path.join(root_dir, img_dir, value["image"])
            if not os.path.exists(image_path):
                print(f"image not found: {image_path}, will be skipped.")
                drop_num += 1
                continue
            for question, answer in zip(value["questions"], value["answers"]):
                c_data = {
                    "datatype": "normal_qa",
                    "question_id": "%06d" %item_num,
                    "metadata": {
                        "question": question,
                        "answer": answer,
                        "type": value["genre"]
                    }
                }
                if image_path not in all_data:
                    all_data[image_path] = []
                all_data[image_path].append(c_data)
                item_num += 1
    all_data = [{"image_path": key, "json": value} for key, value in all_data.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    img_dir = "raw/OCR-VQA/amazon_images_folder"
    for mode in ["train", "val", "test"]:
        save_dir = os.path.join(root_dir, f"processed/{DATASET_NAWE}/{mode}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        process_data(root_dir, save_dir, img_dir, mode)