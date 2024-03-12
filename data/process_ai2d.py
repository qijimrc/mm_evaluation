import os
import json
import tqdm
from utils import save_data

DATASET_NAME = "AI2D"


def process_data(root_dir, mode):
    img_dir = os.path.join(root_dir, mode, "images")
    question_dir = os.path.join(root_dir, mode, "questions")
    save_dir = os.path.join(root_dir, f"processed/{mode}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_results = {}
    for image_file in tqdm.tqdm(os.listdir(img_dir)):
        image_path = os.path.join(img_dir, image_file)
        question_file = image_file + ".json"
        question_path = os.path.join(question_dir, question_file)

        if not os.path.exists(image_path) or not os.path.exists(question_path):
            continue

        with open(question_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        for question_text, question_data in data["questions"].items():
            question_id = question_data["questionId"]
            c_data = {
                "datatype": "multichoice",
                "question_id": question_id,
                "metadata": {
                    "question": question_text,
                    "choices": question_data["answerTexts"],
                    "answer": question_data["correctAnswer"]
                }
            }

            unique_key = f"{image_file}-{question_id}"
            all_results[unique_key] = {"image_path": image_path, "json": c_data}

    # Save data
    all_data = list(all_results.values())
    save_data(all_data, save_dir, DATASET_NAME, mode)


if __name__ == "__main__":
    root_dir = "/Users/zr/Dataset/AI2D"
    mode = "train"  # Processing only the 'train' folder
    print(f"Processing {mode} mode for AI2D dataset.")
    process_data(root_dir, mode)
