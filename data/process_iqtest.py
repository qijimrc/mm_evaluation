import os
import json
import tqdm
import random
from utils import get_image_bytes, save_data

DATASET_NAME = "IQTest"


def answer_to_index(answer):
    return ord(answer.upper()) - ord('A')


def process_data(root_dir, mode):
    img_dir = os.path.join(root_dir, mode, "img")
    label_dir = os.path.join(root_dir, mode, "labels")
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAME}/{mode}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    all_results = {}
    drop_num, item_num = 0, 0

    for label_file in tqdm.tqdm(label_files):
        label_path = os.path.join(label_dir, label_file)
        image_filename = label_file.replace('.json', '.png')
        image_path = os.path.join(img_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
            continue

        with open(label_path, "r", encoding="utf-8") as fp:
            sub_data = json.load(fp)

        if image_path not in all_results:
            all_results[image_path] = []

        answer_index = answer_to_index(sub_data["answer"].strip("()"))

        c_data = {
            "datatype": "multichoice",
            "question_id": "%09d" % item_num,
            "metadata": {
                "question": sub_data["question"],
                "choices": sub_data["choices"],
                "answer": answer_index,
                # "answer_type": "None span",
                # "explanation": sub_data["explanation"]
            }
        }
        all_results[image_path].append(c_data)
        item_num += 1

    # Save tarfiles
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAME, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")


if __name__ == "__main__":
    root_dir = "IQTest"  # Change this to your dataset path

    # Automatically find and process each subfolder
    for mode in os.listdir(root_dir):
        mode_path = os.path.join(root_dir, mode)
        if os.path.isdir(mode_path) and 'img' in os.listdir(mode_path) and 'labels' in os.listdir(mode_path):
            print(f"Processing {mode}.")
            process_data(root_dir, mode)
        else:
            print(f"Skipping {mode} as it does not contain the required 'img' and 'labels' directories.")
