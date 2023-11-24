import os
import json
import random

from utils import save_data

DATASET_NAWE = "VG_box2caption"

def process_data(root_dir, mode):
    img_dir = os.path.join(f"/mnt/shared/img_datasets/shikra/VG_100K_images")
    filename = os.path.join(f"/mnt/shared/img_datasets/shikra/GC_genome196_{mode}.jsonl")
    save_dir = os.path.join(root_dir, f"processed/VG_box2caption/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_results = {}
    drop_num, item_num = 0, 0
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp:
            c_data = json.loads(line)
            image_path = os.path.join(img_dir, *c_data["img_path"].split('/')[1:])
            if not os.path.exists(os.path.join(root_dir, image_path)):
                print(f"not found: {image_path}")
                drop_num += 1
                continue
            if image_path not in all_results:
                all_results[image_path] = []
            c_data = {
                "datatype": "grounding_qa",
                "question_id": item_num,
                "metadata": {
                    "question": "<ph_st><ph_ed>",
                    "question_boxes": [[0]],
                    "answer": c_data['expression'],
                    "answer_boxes": [],
                    "boxes": [c_data["bbox"]],
                    "question_type": "box2caption"
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
    root_dir = "/mnt/shared/img_datasets/grounding_stage2"
    for mode in ['train']:
        print(f"process {mode}.")
        process_data(root_dir, mode)