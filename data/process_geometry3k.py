import os
import json
import tqdm
import random
from utils import save_data

DATASET_NAME = "Geometry3K"

def answer_to_index(answer):
    # 将 A, B, C, D, ... 转换为 0, 1, 2, 3, ...
    return ord(answer.upper()) - ord('A')

def process_data(root_dir, mode):
    mode_dir = os.path.join(root_dir, mode)
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAME}/{mode}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_results = {}
    drop_num, item_num = 0, 0

    for folder in tqdm.tqdm(os.listdir(mode_dir)):
        folder_path = os.path.join(mode_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        data_path = os.path.join(folder_path, "data.json")
        image_path = os.path.join(folder_path, "img_diagram.png")

        if not os.path.exists(data_path) or not os.path.exists(image_path):
            drop_num += 1
            continue

        with open(data_path, "r", encoding="utf-8") as fp:
            sub_data = json.load(fp)

        answer_index = answer_to_index(sub_data["answer"])

        c_data = {
            "datatype": "multichoice",
            "question_id": "%09d" % item_num,
            "metadata": {
                "question": sub_data["problem_text"],
                "choices": sub_data["choices"],
                "answer": answer_index
            }
        }

        if image_path not in all_results:
            all_results[image_path] = []

        all_results[image_path].append(c_data)
        item_num += 1

    # Save tarfiles
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAME, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")


if __name__ == "__main__":
    root_dir = "/Users/zr/Dataset/Geometry3K"  # Change this to your dataset path

    for mode in ["test", "train", "val"]:
        print(f"Processing {mode}.")
        process_data(root_dir, mode)
