import io
import os
import json
import webdataset as wds

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

def process_data(raw_dir, save_dir, img_dir, mode):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    drop_num, result_tar, tar_id, item_num = 0, [], 0, 0
    with open(os.path.join(raw_dir, "data/scienceqa/problems.json"), "r") as fp:
        data = json.load(fp)
        for key, value in tqdm(data.items()):
            if value["split"] != mode:
                continue
            ex_types = []
            train_type = get_eval_type_in_context(value["image"], value["hint"])
            ex_types.extend(get_eval_type_in_subject(value["subject"]))
            ex_types.extend(train_type)
            ex_types.extend(get_eval_type_in_grade(value["grade"]))
            c_tar = {
                "__key__": "%06d" % item_num,
                "prompt": generate_prompt_in_multi_choice(value["choices"], value["question"]),
                "answer": chr(ord('A') + value["answer"]),
                "context": value["hint"],
                "ttype": '$$$'.join(train_type),
                "etype": '$$$'.join(ex_types)
            }
            image_path = None
            if value["image"]:
                image_path = os.path.join(img_dir, key, value["image"])
                if not os.path.exists(image_path):
                    print(f"image not found: {image_path}, will be skipped.")
                    drop_num += 1
                    continue
                img = Image.open(image_path).convert('RGB')
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="jpeg")
                c_tar["jpg"] = img_bytes.getvalue()
            result_tar.append(c_tar)
            item_num += 1
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
        print(f"Save: {item_num}. Drop: {drop_num}")

if __name__ == "__main__":
    raw_dir = "/nxchinamobile2/shared/mmbench_datasets/ScienceQA/raw"
    save_dir = "/nxchinamobile2/shared/mmbench_datasets/ScienceQA/web_dataset"
    for mode in ["train", "val", "test"]:
        img_dir = os.path.join(raw_dir, mode)
        tmp_save_dir = os.path.join(save_dir, mode)
        process_data(raw_dir, tmp_save_dir, img_dir, mode)