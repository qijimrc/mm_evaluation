# import os
#
# import numpy as np
# import pandas as pd
# import json
# from PIL import Image
# import io
#
# def save_image(data, path):
#     image = Image.open(io.BytesIO(data))
#     image.save(path)
#
# def save_json(data, path):
#     with open(path, 'w') as json_file:
#         json.dump(data, json_file)
#
# def process_parquet_file(file_path, output_dir):
#     df = pd.read_parquet(file_path)
#     img_dir = os.path.join(output_dir, 'img')
#     label_dir = os.path.join(output_dir, 'label')
#
#     os.makedirs(img_dir, exist_ok=True)
#     os.makedirs(label_dir, exist_ok=True)
#
#     for index, row in df.iterrows():
#         file_name = f"{index}"
#         # Save image
#         if 'image' in row and 'bytes' in row['image']:
#             img_path = os.path.join(img_dir, file_name + '.jpg')
#             save_image(row['image']['bytes'], img_path)
#
#         # Prepare selected fields, ensuring all data is JSON serializable
#         selected_fields = {
#             'question_id': row['question_id'],
#             'question': row['question'],
#             'choices': list(row['choices']) if isinstance(row['choices'], np.ndarray) else row['choices'],
#             'correct_choice_idx': row['correct_choice_idx']
#         }
#         label_path = os.path.join(label_dir, file_name + '.json')
#         save_json(selected_fields, label_path)
#
# # Directory containing the parquet files
# dataset_dir = '/Users/zr/Dataset/A-OKVQA/'
# files = os.listdir(dataset_dir)
#
# for file in files:
#     if file.endswith('.parquet'):
#         file_path = os.path.join(dataset_dir, file)
#         output_dir = os.path.join(dataset_dir, os.path.splitext(file)[0])
#         process_parquet_file(file_path, output_dir)




import os
import json
import tqdm
import random
from utils import get_image_bytes, save_data

DATASET_NAME = "A-OKVQA"


def answer_to_index(answer):
    return ord(answer.upper()) - ord('A')


def process_data(root_dir, mode):
    img_dir = os.path.join(root_dir, mode, "img")
    label_dir = os.path.join(root_dir, mode, "label")
    save_dir = os.path.join(root_dir, f"processed/{DATASET_NAME}/{mode}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    all_results = {}
    drop_num, item_num = 0, 0

    for label_file in tqdm.tqdm(label_files):
        label_path = os.path.join(label_dir, label_file)
        image_filename = label_file.replace('.json', '.jpg')
        image_path = os.path.join(img_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
            continue

        with open(label_path, "r", encoding="utf-8") as fp:
            sub_data = json.load(fp)

        if image_path not in all_results:
            all_results[image_path] = []

        c_data = {
            "datatype": "multichoice",
            "question_id": "%09d" % item_num,
            "metadata": {
                "question": sub_data["question_id"],
                "choices": sub_data["choices"],
                "answer": sub_data["correct_choice_idx"],
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
    root_dir = "/Users/zr/Dataset/A-OKVQA/A-OKVQA"  # Change this to your dataset path

    # Automatically find and process each subfolder
    for mode in os.listdir(root_dir):
        mode_path = os.path.join(root_dir, mode)
        if os.path.isdir(mode_path) and 'img' in os.listdir(mode_path) and 'label' in os.listdir(mode_path):
            print(f"Processing {mode}.")
            process_data(root_dir, mode)
        else:
            print(f"Skipping {mode} as it does not contain the required 'img' and 'label' directories.")
