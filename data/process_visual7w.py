import os
import random
import math
import json

from PIL import Image
from utils import save_data

DATASET_NAWE = "Visual7W"

def get_box(b):
    return max(b['x'], 0), max(b['y'], 0), b['x']+b['width'], b['y']+b['height']

def process_data(root_dir, mode):
    data_file = os.path.join(root_dir, 'raw/Visual7W/dataset_v7w_pointing.json')
    save_dir = os.path.join(root_dir, f"processed/Visual7W/{mode}")
    image_dir = os.path.join(root_dir, "raw/Visual7W/images")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(data_file) as f:
        data = json.load(f)
    # box id -> box
    box = data['boxes']
    key2box = {}
    for b in box:
        key2box[b['box_id']] = get_box(b)
    all_data, item_num, drop_num = {}, 0, 0
    for img in data['images']:
        if img['split'] != mode:
            continue
        image_path = os.path.join(image_dir, img["filename"])
        if not os.path.exists(image_path):
            drop_num += 1
            print(f'not found {image_path}.')
            continue
        for qa in img["qa_pairs"]:
            boxes = []
            boxes.append(key2box[qa['answer']])
            for q in qa['multiple_choices']:
                boxes.append(key2box[q])
            permute = [0, 1, 2, 3]
            random.shuffle(permute)
            rand_boxes = [boxes[i] for i in permute]
            answer = permute.index(0)
            c_data = {
                "datatype": "grounding_choice",
                "question_id": qa["qa_id"],
                "metadata": {
                    "question": qa["question"],
                    "choices": [f'<ph_st><ph_ed>' for _ in permute],
                    "answer": answer,
                    "question_boxes": [],
                    "choices_boxes": [[i] for i in range(len(permute))],
                    "boxes": rand_boxes,
                    "type": qa["type"]
                }
            }
            if image_path not in all_data:
                all_data[image_path] = []
            all_data[image_path].append(c_data)
            item_num += 1
    # save tarfiles
    all_data = [{"image_path": key, "json": value} for key, value in all_data.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")
    

if __name__ == '__main__':
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ["test", "val", "train"]:
        print(f"process {mode}.")
        process_data(root_dir, mode)