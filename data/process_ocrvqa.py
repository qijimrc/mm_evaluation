import io
import os
import json
import tqdm
import webdataset as wds

from PIL import Image
from utils import get_image_bytes

def process_data(raw_dir, save_dir, img_dir, mode):
    drop_num, result_tar, tar_id, item_num = 0, [], 0, 0
    type_dict = {1: "train", 2: "val", 3: "test"}
    all_data = {}
    with open(os.path.join(root_dir, "raw/OCR-VQA/dataset.json"), "r") as fp:
        data = json.load(fp)
        for key, value in tqdm.tqdm(data.items()):
            if type_dict[value["split"]] != mode:
                continue
            image_path = os.path.join(img_dir, value["image"])
            if not os.path.exists(image_path):
                print(f"image not found: {image_path}, will be skipped.")
                drop_num += 1
                continue
            for question, answer in zip(value["questions"], value["answers"]):
                c_data = {
                    "question_id": "%06d" %item_num,
                    "prompt": question,
                    "answer": answer,
                    "type": value["genre"]
                }
                if image_path not in all_data:
                    all_data[image_path] = []
                all_data[image_path].append(c_data)
                item_num += 1
    # save tarfiles
    tar_id, result_tar, image_num = 0, [], 0
    for key, value in tqdm.tqdm(all_data.items()):
        c_tar = {
            "__key__": "%06d" %image_num,
            "json": value,
            "jpg": get_image_bytes(key)
        }
        result_tar.append(c_tar)
        image_num += 1
        if len(result_tar) >= 1000:
            with wds.TarWriter(os.path.join(save_dir, f"{mode}_ocrvqa_%06d.tar" %(tar_id)), "w") as tar:
                for res in result_tar:
                    tar.write(res)
            result_tar = []
            tar_id += 1
    if len(result_tar) > 0:
        with wds.TarWriter(os.path.join(save_dir, f"{mode}_ocrvqa_%06d.tar" %(tar_id)), "w") as tar:
            for res in result_tar:
                tar.write(res)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    img_dir = os.path.join(root_dir, "raw/OCR-VQA/amazon_images_folder")
    for mode in ["train", "val", "test"]:
        save_dir = os.path.join(root_dir, f"processed/OCR-VQA/{mode}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        process_data(root_dir, save_dir, img_dir, mode)