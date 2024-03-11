import os
import json
import tqdm
from utils import save_data

DATASET_NAME = "Tabmwp"


def answer_to_index(choices, answer):
    try:
        return choices.index(answer)
    except ValueError:
        return -1


def process_json_file(root_dir, json_file_path, mode):
    save_dir = os.path.join(root_dir, f"processed/{mode}")
    tables_dir = os.path.join(root_dir, "tables")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(json_file_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    all_results = {}
    for key, value in tqdm.tqdm(data.items()):
        image_filename = f"{key}.png"

        image_path = os.path.join(tables_dir, image_filename)

        if not os.path.exists(image_path):
            continue

        datatype = "multichoice" if value["ques_type"] == "multi_choice" else "normal_qa"
        metadata = {
            "question": value["question"],
        }

        if datatype == "multichoice":
            metadata["choices"] = value["choices"]
            metadata["answer"] = answer_to_index(value["choices"], value["answer"])
        else:
            metadata["answer"] = value["answer"]

        c_data = {
            "datatype": datatype,
            "question_id": key,
            "metadata": metadata
        }

        if image_path not in all_results:
            all_results[image_path] = []

        all_results[image_path].append(c_data)

    # Save data
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    save_data(all_data, save_dir, DATASET_NAME, mode)


if __name__ == "__main__":
    root_dir = "/Users/zr/Dataset/Tabmwp"  # Change this to your dataset path
    for json_file in os.listdir(root_dir):
        if json_file.startswith("problems_") and json_file.endswith(".json"):
            mode = json_file.replace("problems_", "").replace(".json", "")
            json_file_path = os.path.join(root_dir, json_file)
            print(f"Processing {mode}.")
            process_json_file(root_dir, json_file_path, mode)
