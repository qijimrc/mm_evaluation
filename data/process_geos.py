import os
import json
import tqdm
import random
from utils import save_data

DATASET_NAME = "GEOS"

def process_data(root_dir, mode):
    img_dir = os.path.join(root_dir, mode)
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAME}/{mode}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    label_files = [f for f in os.listdir(img_dir) if f.endswith('.json')]
    all_results = {}
    drop_num, item_num = 0, 0

    for label_file in tqdm.tqdm(label_files):
        label_path = os.path.join(img_dir, label_file)
        image_filename = label_file.replace('.json', '.png')  # Assuming image files are .png
        image_path = os.path.join(img_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
            continue

        with open(label_path, "r", encoding="utf-8") as fp:
            sub_data = json.load(fp)

        # 将答案转换为选项下标
        answer_index = int(sub_data["answer"]) - 1

        c_data = {
            "datatype": "multichoice",
            "question_id": "%09d" % item_num,
            "metadata": {
                "question": sub_data["text"],
                "choices": list(sub_data["choices"].values()),
                "answer": answer_index,
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
    root_dir = "/Users/zr/Dataset/GEOS"  # Change this to your dataset path

    for mode in ["aaai", "official", "practice"]:
        print(f"Processing {mode}.")
        process_data(root_dir, mode)
