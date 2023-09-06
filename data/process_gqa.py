import os
import json
import glob
import jsonlines
from tqdm import tqdm
import pandas as pd
import webdataset as wds

from utils import get_image_bytes

def merge_jsonl_to_single(filename_list, root_dir, img_dir, save_filename):
    im2data = {}
    drop_num = 0
    question_id = 0
    for filename in filename_list:
        with open(filename, "r", encoding='utf-8') as f:
            processed_datas = list(json.load(f).values())
        print(f"process {len(processed_datas)} samples in {filename}")
        for i, data in tqdm(enumerate(processed_datas)):
            if 'imageId' not in data or 'question' not in data or 'answer' not in data  or 'fullAnswer' not in data:
                drop_num += 1
                continue
            image_path = os.path.join(root_dir, img_dir, data['imageId'] + ".jpg")
            if not os.path.exists(image_path):
                drop_num += 1
                continue
            prompt, txt, full_answer = data['question'], data['answer'], data['fullAnswer']
            conversation = {
                "question_id": "%09d" %(question_id),
                "prompt":prompt,
                "txt": txt,
                "full_answer": full_answer
            }
            question_id += 1
            image_id = data["imageId"]
            if image_id not in im2data:
                im2data[image_id] = []
            im2data[image_id].append(conversation)

    total_question = sum(len(v) for k,v in im2data.items())
    print(f"find {len(im2data)} images, {total_question} questions\n drop {drop_num} invalid questions")
    with open(save_filename, 'w', encoding='utf-8') as f:
        for img_id, conversations in im2data.items():
            img_path = os.path.join(root_dir, img_dir, img_id + ".jpg")
            data = {"image_path": img_path, "conversation": conversations}
            f.write(json.dumps(data, ensure_ascii=False)+"\n")

def process_data_to_wds(filename, save_dir, mode):
    tar_id = 0
    result_tar = []
    item_num = 0
    with jsonlines.open(filename, 'r') as fp:
        for data in tqdm(fp):
            c_tar = {
                "__key__": "%09d" %(item_num),
                "jpg": get_image_bytes(data["image_path"]),
                "json": data["conversation"],
            }
            result_tar.append(c_tar)
            item_num += 1
            if len(result_tar) >= 1000:
                with wds.TarWriter(os.path.join(save_dir, f"GQA_{mode}_%06d.tar" %(tar_id)), "w") as tar:
                    for res in result_tar:
                        tar.write(res)
                    tar_id += 1
                    result_tar = []
        if len(result_tar) > 0:
            with wds.TarWriter(os.path.join(save_dir, f"GQA_{mode}_%06d.tar" %(tar_id)), "w") as tar:
                for res in result_tar:
                    tar.write(res)
        print(f"save {item_num} samples.")


if __name__ == "__main__":
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    img_dir = "GQA/images"
    for mode in ["train", "val", "test"]:
        print(f'process {mode}')
        save_dir = os.path.join(root_dir, f"processed/GQA/{mode}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # merge image data
        if mode == "train":
            file_list = glob.glob(os.path.join(root_dir, f"GQA/train_all_questions/*.json"))
        elif mode == "val":
            file_list = [os.path.join(root_dir, f"GQA/val_all_questions.json")]
        else:
            file_list = [os.path.join(root_dir, f"GQA/testdev_all_questions.json")]
        save_filename = os.path.join(save_dir, f"{mode}.jsonl")
        merge_jsonl_to_single(file_list, root_dir, img_dir, save_filename)
        # save wds
        process_data_to_wds(save_filename, save_dir, mode)
