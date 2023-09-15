import io
import os
import json
import random
import pandas as pd
import webdataset as wds

from PIL import Image
from tqdm import tqdm
from utils import generate_prompt_in_multi_choice, get_image_bytes, save_data

DATASET_NAWE = "ScienceQA"

def get_eval_type_in_context(image, hint):
    ret = []
    if image is None and len(hint) <= 0:
        return ["NO"]
    if image:
        ret.append("IMG")
    if len(hint) > 0:
        ret.append("TXT")
    return ret

def get_eval_type_in_subject(subject):
    topic_map = {
        "language science": "LAN",
        "natural science": "NAT",
        "social science": "SOC"
    }
    return [topic_map[subject]]
    
def get_eval_type_in_grade(grade):
    if grade in set(["grade1", "grade2", "grade3", "grade4", "grade5", "grade6"]):
        return ["G1-6"]
    if grade in set(["grade7", "grade8", "grade9", "grade10", "grade11", "grade12"]):
        return ["G7-12"]
    raise ValueError("Invalid grade: %s" % grade)

def process_data(root_dir, save_dir, img_dir, mode):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    drop_num, item_num = 0, 0
    all_results = {}
    with open(os.path.join(root_dir, "raw/ScienceQA/data/scienceqa/problems.json"), "r") as fp:
        data = json.load(fp)
        for key, value in tqdm(data.items()):
            if value["split"] != mode:
                continue
            ex_types = []
            train_type = get_eval_type_in_context(value["image"], value["hint"])
            ex_types.extend(get_eval_type_in_subject(value["subject"]))
            ex_types.extend(train_type)
            ex_types.extend(get_eval_type_in_grade(value["grade"]))
            image_path = f"$$$-{key}"
            if value["image"]:
                image_path = os.path.join(root_dir, img_dir, key, value["image"])
                if not os.path.exists(image_path):
                    print(f"image not found: {image_path}, will be skipped.")
                    drop_num += 1
                    continue
            if image_path not in all_results:
                all_results[image_path] = []
            for ttype in train_type:
                c_res = {
                    "datatype": "multichoice",
                    "question_id": "%09d" %item_num,
                    "metadata": {
                        "question": value["question"],
                        "choices": value["choices"],
                        "answer": value["answer"],
                        "ttype": ttype,
                        "etype": '$$$'.join(ex_types)
                    }
                }
                # c_res = {
                #     "question_id": item_num, 
                #     "prompt": generate_prompt_in_multi_choice(value["choices"], value["question"]), 
                #     "answer": chr(ord('A') + value["answer"]), 
                #     "context": value['hint'] if value['hint'] else "",
                #     "ttype": ttype, 
                #     "etype": ex_types
                # }
                all_results[image_path].append(c_res)
                item_num += 1
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ["train", "val", "test"]:
        print(f"Start {mode}.")
        img_dir = os.path.join(root_dir, "raw/ScienceQA", mode)
        save_dir = os.path.join(root_dir, f"processed/ScienceQA/{mode}")
        process_data(root_dir, save_dir, img_dir, mode)