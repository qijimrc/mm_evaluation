import os
import json
import tqdm
import random
from utils import save_data

DATASET_NAME = "DVQA"

def process_data(root_dir, mode):
    img_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAME}/{mode}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    label_file = os.path.join(label_dir, f"{mode}_qa.json")
    all_data = []
    drop_num, item_num = 0, 0

    with open(label_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    for entry in tqdm.tqdm(data):
        image_filename = entry["image"]
        image_path = os.path.join(img_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
            continue

        single_question_record = {
            "key": image_path.split('/')[-1].split('.')[0],  # Extracting key from the image filename
            "json": [{
                "datatype": "normal_qa",
                "question_id": entry["question_id"],
                "metadata": {
                    "question": entry["question"],
                    "answer": entry["answer"],
                }
            }],
            "image_path": image_path
        }

        all_data.append(single_question_record)
        item_num += 1

    # Save records
    random.shuffle(all_data)
    save_data(all_data, save_dir, DATASET_NAME, mode)
    print(f"Saved: {len(all_data)} records. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/Users/zr/Dataset/DVQA"
    for mode in ["train", "val_easy", "val_hard"]:
        print(f"Processing {mode}.")
        process_data(root_dir, mode)
