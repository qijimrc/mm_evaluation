import os
import io
import shutil
import jsonlines
from tqdm import tqdm
from PIL import Image
import webdataset as wds

PROMPT_EN = "Please choose the correct option for the above question from the following options: "
PROMPT_ZH = "请从以下几个选项中选出上述问题的正确答案："

def generate_prompt_in_multi_choice(choices, question, language="zh"):
    prompt = question + "\n" + (PROMPT_ZH if language == "zh" else PROMPT_EN) + "\n"
    start_op = 'A'
    for item in choices:
        prompt += f'{start_op}: {item}\n'
        start_op = chr(ord(start_op) + 1)
    return prompt

def get_image_bytes(image_path, img_save_path):
    if type(image_path) is not str:
        data = image_path
    else:
        with open(image_path, "rb") as f:
            data = f.read()
    with open(img_save_path, "wb") as f:
        f.write(data)
    return data
    
def save_files(save_dir, filename, result_tar, result_meta):
    with wds.TarWriter(os.path.join(save_dir, f'{filename}.tar'), "w") as tar:
        for res in result_tar:
            tar.write(res)
    with jsonlines.open(os.path.join(save_dir, f'{filename}.meta.jsonl'), "w") as jsonl:
        for meta in result_meta:
            jsonl.write(meta)
            
def save_data(all_data, save_dir, dataset_name, mode):
    # 保存数据
    file_id, result_tar, result_meta, image_num = 0, [], [], 0
    for data in tqdm(all_data):
        c_filename = f"{dataset_name}_{mode}_%06d" %file_id
        if not os.path.exists(os.path.join(save_dir, c_filename)):
            os.mkdir(os.path.join(save_dir, c_filename))

        join_id = "%09d" %(image_num)
        c_tar = {
            "__key__": join_id,
            "id": join_id,
        }
        c_meta = {
            "key": join_id,
            "json": data["json"]
        }
        c_meta["image_path"] = os.path.join(os.path.join(c_filename, f'{join_id}.jpg'))
        if "image_path" in data and not data["image_path"].startswith("$$$"):
            c_tar["jpg"] = get_image_bytes(data["image_path"], os.path.join(save_dir, c_meta["image_path"]))
        else:
            c_tar['jpg'] = get_image_bytes(data["image_bytes"], os.path.join(save_dir, c_meta["image_path"]))

        result_meta.append(c_meta)
        result_tar.append(c_tar)
        image_num += 1
        if len(result_tar) >= 1000:
            save_files(save_dir, c_filename, result_tar, result_meta)
            file_id += 1
            result_tar, result_meta = [], []

    if len(result_tar) > 0:
        save_files(save_dir, c_filename, result_tar, result_meta)
    return image_num