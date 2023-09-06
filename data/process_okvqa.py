import os
import json
import tqdm
import pandas as pd
import webdataset as wds
from collections import Counter
from utils import get_image_bytes

def select_answer_by_confidence(answers):
    confidenced_answers = [answer["answer"] for answer in answers if answer["answer_confidence"] == "yes"]
    if len(confidenced_answers) == 0:
        return None
    counts = Counter(confidenced_answers)
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return counts[0][0]

def process_data(root_dir, save_dir, img_dir, mode):
    okvqa_questions_f = os.path.join(root_dir, f"raw/OK-VQA/{mode}/OpenEnded_mscoco_{mode}2014_questions.json")
    okvqa_anns_f = os.path.join(root_dir, f"raw/OK-VQA/{mode}/mscoco_{mode}2014_annotations.json")
    with open(okvqa_questions_f) as f1, open(okvqa_anns_f) as f2:
        questions, anns = json.load(f1), json.load(f2)
    
    sorted_qs = sorted(questions['questions'], key=lambda x:x['question_id'])
    sorted_anns = sorted(anns['annotations'], key=lambda x:x['question_id'])

    all_data, drop_num, item_num = {}, 0, 0
    for idx in tqdm.tqdm(range(len(sorted_qs))):
        q_info, a_info = sorted_qs[idx], sorted_anns[idx]
        assert q_info["image_id"] == a_info["image_id"] and q_info["question_id"] == a_info["question_id"]
        qa_info = q_info | a_info
        img_path = os.path.join(img_dir, 'COCO_{}2014_{}{}.jpg'.format(mode, ''.join(['0']*(12-len(str(qa_info['image_id'])))), qa_info['image_id']))
        if not os.path.exists(img_path):
            drop_num += 1
            print(f'not found {img_path}.')
            continue
        answer = select_answer_by_confidence(qa_info["answers"])
        if answer is None:
            drop_num += 1
            print(f'no confidenced answer!')
            continue
        c_data = {k: qa_info[k] for k in ["question_id", "question_type", "question", "answer_type"]}
        c_data["answer"] = answer
        if img_path not in all_data:
            all_data[img_path] = []
        all_data[img_path].append(c_data)
        item_num += 1
    # save tarfiles
    tar_id, result_tar, image_num = 0, [], 0
    for key, value in tqdm.tqdm(all_data.items()):
        c_tar = {
            "__key__": "%06d" %image_num,
            "json": value,
            "jpg": get_image_bytes(key)
        }
        result_tar.append(c_tar)
        image_num += 1
        if len(result_tar) >= 1000:
            with wds.TarWriter(os.path.join(save_dir, f"{mode}_okvqa_%06d.tar" %(tar_id)), "w") as tar:
                for res in result_tar:
                    tar.write(res)
            result_tar = []
            tar_id += 1
    if len(result_tar) > 0:
        with wds.TarWriter(os.path.join(save_dir, f"{mode}_okvqa_%06d.tar" %(tar_id)), "w") as tar:
            for res in result_tar:
                tar.write(res)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ["train", "val"]:
        print(f'process {mode}')
        img_dir = os.path.join(root_dir, f"raw/OK-VQA/images/{mode}2014")
        save_dir = os.path.join(root_dir, f"processed/OK-VQA/{mode}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        process_data(root_dir, save_dir, img_dir, mode)
