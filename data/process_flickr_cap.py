import glob
import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter
from utils import get_image_bytes, save_data


# with open('templates.json') as f:
#     templates = json.load(f)
#     cap_template = templates['Caption']

DATASET_NAME = "FlickrCap"

def process_data(root_dir, mode):
    anns_file = os.path.join(root_dir, f'raw/flickr/dataset_flickr30k.json')
    img_dir = f"/nxchinamobile2/shared/img_datasets/shikra/flickr30k-images"
    urls = glob.glob(img_dir + '/*')
    name2url = {url.split('/')[-1]: url for url in urls}
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAME}/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(anns_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
        annotations = data['images']

    all_results = {}
    drop_num, item_num = 0, 0
    for ann in tqdm.tqdm(annotations):
        if mode not in ann['split']:
            continue
        image_path = os.path.join(img_dir, ann['filename'])
        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
        if image_path not in all_results:
            all_results[image_path] = []
        sentences = ann['sentences']
        for sen in sentences:
            answer = sen['raw']
            c_data = {
                "datatype": "normal_caption",
                "question_id": "%09d" %item_num,
                "metadata": {
                    "answer": answer,
                }
            }
            all_results[image_path].append(c_data)
            item_num += 1
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAME, "test")
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ['train', 'val', 'test']:
        process_data(root_dir, mode)
