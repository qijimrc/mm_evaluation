import io
import os
import json
import pandas as pd

from PIL import Image
from tqdm import tqdm
from utils import generate_prompt_in_multi_choice

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
        os.mkdir(save_dir)
    drop_num, results, item_num = 0, [], 0
    with open(os.path.join(root_dir, "ScienceQA/raw/data/scienceqa/problems.json"), "r") as fp:
        data = json.load(fp)
        for key, value in tqdm(data.items()):
            if value["split"] != mode:
                continue
            ex_types = []
            train_type = get_eval_type_in_context(value["image"], value["hint"])
            ex_types.extend(get_eval_type_in_subject(value["subject"]))
            ex_types.extend(train_type)
            ex_types.extend(get_eval_type_in_grade(value["grade"]))
            image_path = ""
            if value["image"]:
                image_path = os.path.join(img_dir, key, value["image"])
                if not os.path.exists(os.path.join(root_dir, image_path)):
                    print(f"image not found: {image_path}, will be skipped.")
                    drop_num += 1
                    continue
            context = value['hint'] if value['hint'] else ""
            for ttype in train_type:
                results.append([item_num, # question_id
                                image_path, # image_path
                                generate_prompt_in_multi_choice(value["choices"], value["question"]), # prompt
                                chr(ord('A') + value["answer"]), # answer
                                context, # context
                                ttype, # ttype,
                                '$$$'.join(ex_types) # etype
                                ])
            item_num += 1
            
        df = pd.DataFrame(results, columns=["question_id", "image_path", "prompt", "answer", "context", "ttype", "etype"])
        df.to_csv(os.path.join(save_dir, f"{mode}.csv"), index=False)
        print(f"Save: {item_num}. Drop: {drop_num}")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    save_dir = os.path.join(root_dir, "ScienceQA/csv_files")
    for mode in ["train", "val", "test"]:
        print(f"Start {mode}.")
        img_dir = os.path.join("ScienceQA/raw", mode)
        process_data(root_dir, save_dir, img_dir, mode)