import os
import json
import tqdm
import random
import pandas as pd
import webdataset as wds
from collections import Counter
from utils import get_image_bytes, save_data

DATASET_NAWE = "OKVQA"

def select_answer_by_confidence(answers):
    answer_list = [answer["answer"] for answer in answers]
    if len(answer_list) == 0:
        return None, None
    counts = Counter(answer_list)
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return counts[0][0], answer_list

def process_data(root_dir, save_dir, img_dir, mode):
    okvqa_questions_f = os.path.join(root_dir, f"raw/OK-VQA/{mode}/OpenEnded_mscoco_{mode}2014_questions.json")
    okvqa_anns_f = os.path.join(root_dir, f"raw/OK-VQA/{mode}/mscoco_{mode}2014_annotations.json")
    with open(okvqa_questions_f) as f1, open(okvqa_anns_f) as f2:
        questions, anns = json.load(f1), json.load(f2)
    
    sorted_qs = sorted(questions['questions'], key=lambda x:x['question_id'])
    sorted_anns = sorted(anns['annotations'], key=lambda x:x['question_id'])
    question_types = anns["question_types"]

    all_data, drop_num, item_num = {}, 0, 0
    for idx in tqdm.tqdm(range(len(sorted_qs))):
        q_info, a_info = sorted_qs[idx], sorted_anns[idx]
        assert q_info["image_id"] == a_info["image_id"] and q_info["question_id"] == a_info["question_id"]
        qa_info = q_info | a_info
        img_path = os.path.join(root_dir, img_dir, 'COCO_{}2014_{}{}.jpg'.format(mode, ''.join(['0']*(12-len(str(qa_info['image_id'])))), qa_info['image_id']))
        if not os.path.exists(img_path):
            drop_num += 1
            print(f'not found {img_path}.')
            continue
        answer, answer_list = select_answer_by_confidence(qa_info["answers"])
        if answer is None:
            drop_num += 1
            print(f'no confidenced answer!')
            continue
        c_data = {
            "datatype": "normal_qa",
            "question_id": qa_info["question_id"],
            "metadata": {
                "question": qa_info["question"],
                "answer": answer,
                "answer_list": answer_list,
                "question_type": question_types[qa_info["question_type"]]
            }
        }
        if img_path not in all_data:
            all_data[img_path] = []
        all_data[img_path].append(c_data)
        item_num += 1
    all_data = [{"image_path": key, "json": value} for key, value in all_data.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ["train", "val"]:
        print(f'process {mode}')
        img_dir = f"raw/OK-VQA/images/{mode}2014"
        save_dir = os.path.join(root_dir, f"processed/OK-VQA/{mode}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        process_data(root_dir, save_dir, img_dir, mode)
