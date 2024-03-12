# import os
# import pickle
# from pathlib import Path
# import json
# from PIL import Image
# import numpy as np
#
# def save_image(image_array, image_path):
#     img = Image.fromarray(image_array)
#     img.save(image_path)
#
# def save_labels(item, label_path):
#     label_data = {
#         'question': item.get('subject', item.get('English_problem', '')),
#         'choices': item.get('choices', []),
#         'answer': item.get('label', ''),
#         'explanation': item.get('answer', ''),
#
#     }
#
#     with open(label_path, 'w') as file:
#         json.dump(label_data, file, ensure_ascii=False, indent=4)
#
#
# def process_pk_file(file_path, output_dir):
#     with open(file_path, 'rb') as file:
#         data = pickle.load(file)
#
#     img_dir = output_dir / 'img'
#     labels_dir = output_dir / 'labels'
#     img_dir.mkdir(parents=True, exist_ok=True)
#     labels_dir.mkdir(parents=True, exist_ok=True)
#
#     for i, item in enumerate(data):
#         if 'image' in item:
#             image_path = img_dir / f'{i}.png'
#             save_image(item['image'], image_path)
#             del item['image']  # Remove image data from item
#
#         label_path = labels_dir / f'{i}.json'
#         save_labels(item, label_path)
#
# def main(dataset_dir):
#     dataset_dir = Path(dataset_dir)
#     for pk_file in dataset_dir.glob('*.pk'):
#         output_dir = dataset_dir / pk_file.stem
#         process_pk_file(pk_file, output_dir)
#
# # Replace with your dataset directory
# dataset_directory = '/Users/zr/Dataset/UniGeo_PK'
# main(dataset_directory)


import os
import json
import tqdm
from utils import save_data

DATASET_DIR = "/Users/zr/Dataset/UniGeo/"
DATASET_NAME = "UniGeo"


def process_data(mode):
    img_dir = os.path.join(DATASET_DIR, mode, "img")
    label_dir = os.path.join(DATASET_DIR, mode, "labels")
    save_dir = os.path.join(DATASET_DIR, f"processed/{mode}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    all_results = {}

    for label_file in tqdm.tqdm(label_files):
        label_path = os.path.join(label_dir, label_file)
        image_filename = label_file.replace('.json', '.png')  # Assuming image files are .png
        image_path = os.path.join(img_dir, image_filename)

        if not os.path.exists(image_path):
            continue

        with open(label_path, "r", encoding="utf-8") as fp:
            sub_data = json.load(fp)

        c_data = {
            "datatype": "multichoice",
            "question_id": label_file.replace('.json', ''),
            "metadata": {
                "question": sub_data["question"],
                "choices": sub_data["choices"],
                "answer": sub_data["answer"],
                "explanation": sub_data.get("explanation", "")
            }
        }

        if image_path not in all_results:
            all_results[image_path] = []

        all_results[image_path].append(c_data)

    # Save data
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    save_data(all_data, save_dir, DATASET_NAME, mode)


if __name__ == "__main__":
    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if os.path.isdir(folder_path) and "calculation" in folder or "proving" in folder:
            print(f"Processing {folder}.")
            process_data(folder)
