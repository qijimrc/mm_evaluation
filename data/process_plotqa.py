import os
import json
import tqdm
from utils import save_data
DATASET_NAME = "PlotQA"


def process_data(root_dir, mode):
    img_dir = os.path.join(root_dir, mode, "png")
    qa_file = os.path.join(root_dir, mode, "qa_pairs_V1.json")
    save_dir = os.path.join(root_dir, f"processed/{mode}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(qa_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    all_results = {}
    for qa_pair in tqdm.tqdm(data["qa_pairs"]):
        image_filename = f"{qa_pair['image_index']}.png"
        image_path = os.path.join(img_dir, image_filename)

        if not os.path.exists(image_path):
            continue

        c_data = {
            "datatype": "normal_qa",
            "question_id": qa_pair["question_id"],
            "metadata": {
                "question": qa_pair["question_string"],
                "answer": qa_pair["answer"],
                "type": qa_pair.get("type", ""),
                "question_id": qa_pair["question_id"]
            }
        }

        if image_path not in all_results:
            all_results[image_path] = []

        all_results[image_path].append(c_data)

    # Save data
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    save_data(all_data, save_dir, DATASET_NAME, mode)


if __name__ == "__main__":
    root_dir = "/Users/zr/Dataset/PlotQA"
    for folder in ["test", "train", "val"]:
        print(f"Processing {folder}.")
        process_data(root_dir, folder)
