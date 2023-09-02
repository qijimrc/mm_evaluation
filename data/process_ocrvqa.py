import io
import os
import json
import webdataset as wds

from PIL import Image
from tqdm import tqdm
from utils import get_image_bytes

def process_data(raw_dir, save_dir, img_dir, mode):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    drop_num, result_tar, tar_id, item_num = 0, [], 0, 0
    type_dict = {1: "train", 2: "val", 3: "test"}
    with open(os.path.join(raw_dir, "dataset.json"), "r") as fp:
        data = json.load(fp)
        for key, value in tqdm(data.items()):
            if type_dict[value["split"]] != mode:
                continue
            c_tar = {
                "__key__": key,
                "prompt": value["questions"][0],
                "answer": value["answers"][0],
                "json": {"questions": value["questions"], "answers": value["answers"]},
                "ttype": "IMG",
                "etype": value["genre"]
            }
            if value["image"]:
                image_path = os.path.join(img_dir, value["image"])
                if not os.path.exists(image_path):
                    print(f"image not found: {image_path}, will be skipped.")
                    drop_num += 1
                    continue
                c_tar["jpg"] = get_image_bytes(image_path)
            else:
                print(f"image not found, will be skipped.")
                drop_num += 1
                continue
                
            result_tar.append(c_tar)
            item_num += 1
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
        print(f"Save: {item_num}. Drop: {drop_num}")

if __name__ == "__main__":
    raw_dir = "/nxchinamobile2/shared/mmbench_datasets/OCR-VQA/raw"
    save_dir = "/nxchinamobile2/shared/mmbench_datasets/OCR-VQA/web_dataset"
    img_dir = os.path.join(raw_dir, "amazon_images_folder")
    for mode in ["train", "val", "test"]:
        tmp_save_dir = os.path.join(save_dir, mode)
        process_data(raw_dir, tmp_save_dir, img_dir, mode)