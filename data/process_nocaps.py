import os
import json
import tqdm
import pandas as pd
import webdataset as wds
from collections import Counter
from utils import get_image_bytes


# with open('templates.json') as f:
#     templates = json.load(f)
#     cap_template = templates['Caption']

def process_data(root_dir):
    anns_file = os.path.join(root_dir, f"raw/NoCaps/nocaps_val_4500_captions.json")
    img_dir = os.path.join(root_dir, "raw/NoCaps/imgs")    
    save_dir = os.path.join(root_dir, f"processed/NoCaps/val")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(anns_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
        annotations = data['annotations']
        id2imgs = {info['id']: info for info in data['images']}

    all_results = {}
    drop_num, item_num = 0, 0
    for ann in tqdm.tqdm(annotations):
        imginfo = id2imgs[ann['image_id']]
        image_path = os.path.join(img_dir, imginfo['file_name'])
        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
        if image_path not in all_results:
            all_results[image_path] = []
        all_results[image_path].append({
            "image_id": ann["image_id"],
            "question_id": ann["id"],
            "answer": ann["caption"],
            'open_image_id': imginfo['open_images_id'],
            'domain': imginfo['domain']
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
            with wds.TarWriter(os.path.join(save_dir, f"val_nocaps_%06d.tar" %(tar_id)), "w") as tar:
                for res in result_tar:
                    tar.write(res)
            result_tar = []
            tar_id += 1
    if len(result_tar) > 0:
        with wds.TarWriter(os.path.join(save_dir, f"val_nocaps_%06d.tar" %(tar_id)), "w") as tar:
            for res in result_tar:
                tar.write(res)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    process_data(root_dir)
