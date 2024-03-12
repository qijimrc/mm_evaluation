import os
import json
import tqdm
from utils import save_data

DATASET_NAME = "IconQA"


def process_subfolder(root_dir, mode, subfolder):
    subfolder_dir = os.path.join(root_dir, mode, subfolder)
    save_dir = os.path.join(root_dir, f"processed/{mode}/{subfolder}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_results = {}
    for folder in tqdm.tqdm(os.listdir(subfolder_dir)):
        folder_path = os.path.join(subfolder_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        image_path = os.path.join(folder_path, "image.png")
        data_path = os.path.join(folder_path, "data.json")

        if not os.path.exists(image_path) or not os.path.exists(data_path):
            continue

        with open(data_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        datatype = "multichoice" if subfolder == "choose_txt" else "normal_qa"
        metadata = {
            "question": data["question"],

        }
        if datatype == "multichoice":
            metadata["choices"] = data["choices"]

        if "answer" in data:
            metadata["answer"] = data["answer"]
        c_data = {
            "datatype": datatype,
            "question_id": folder,  # 使用文件夹名称作为问题ID
            "metadata": metadata
        }

        if image_path not in all_results:
            all_results[image_path] = []

        all_results[image_path].append(c_data)

    # Save data
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    save_data(all_data, save_dir, DATASET_NAME, mode)


if __name__ == "__main__":
    root_dir = "/Users/zr/Dataset/IconQA/iconqa"
    for mode in ["test", "val", "train"]:
        for subfolder in ["fill_in_blank", "choose_txt"]:
            print(f"Processing {mode}/{subfolder}.")
            process_subfolder(root_dir, mode, subfolder)
