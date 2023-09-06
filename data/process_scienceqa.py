import io
import os
import json
import pandas as pd
import webdataset as wds

from PIL import Image
from tqdm import tqdm
from utils import generate_prompt_in_multi_choice, get_image_bytes

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
            image_path = "$$$"
            if value["image"]:
                image_path = os.path.join(img_dir, key, value["image"])
                if not os.path.exists(os.path.join(root_dir, image_path)):
                    print(f"image not found: {image_path}, will be skipped.")
                    drop_num += 1
                    continue
            if image_path not in all_results:
                all_results[image_path] = []
            for ttype in train_type:
                c_res = {
                    "question_id": item_num, 
                    "prompt": generate_prompt_in_multi_choice(value["choices"], value["question"]), 
                    "answer": chr(ord('A') + value["answer"]), 
                    "context": value['hint'] if value['hint'] else "",
                    "ttype": ttype, 
                    "etype": ex_types
                }
                all_results[image_path].append(c_res)
            item_num += 1
    # save tarfiles
    image_num, non_image_num = 0, 0
    tar_id, result_tar = 0, []
    if "$$$" in all_results:
        for data in all_results["$$$"]:
            c_tar = {
                "__key__": "%06d" %image_num,
                "json": [data]
            }
            result_tar.append(c_tar)
            image_num += 1
            non_image_num += 1
            if len(result_tar) >= 1000:
                with wds.TarWriter(os.path.join(save_dir, f"{mode}_scienceqa_%06d.tar" %(tar_id)), "w") as tar:
                    for res in result_tar:
                        tar.write(res)
                result_tar = []
                tar_id += 1
    all_results.pop("$$$")
    for key, value in tqdm(all_results.items()):
        c_tar = {
            "__key__": "%06d" %image_num,
            "json": value,
            "jpg": get_image_bytes(key)
        }
        result_tar.append(c_tar)
        image_num += 1
        result_tar.append(c_tar)
        if len(result_tar) >= 1000:
            with wds.TarWriter(os.path.join(save_dir, f"{mode}_scienceqa_%06d.tar" %(tar_id)), "w") as tar:
                for res in result_tar:
                    tar.write(res)
            result_tar = []
            tar_id += 1
    if len(result_tar) > 0:
        with wds.TarWriter(os.path.join(save_dir, f"{mode}_scienceqa_%06d.tar" %(tar_id)), "w") as tar:
            for res in result_tar:
                tar.write(res)
    print(f"Save: {image_num - non_image_num} images, {non_image_num} non images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ["train", "val", "test"]:
        print(f"Start {mode}.")
        img_dir = os.path.join(root_dir, "raw/ScienceQA", mode)
        save_dir = os.path.join(root_dir, f"processed/ScienceQA/{mode}")
        process_data(root_dir, save_dir, img_dir, mode)