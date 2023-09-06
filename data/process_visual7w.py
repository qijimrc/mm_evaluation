import os
import random
import math
import json
import tqdm
import webdataset as wds

from PIL import Image
from sat.helpers import print_rank0
from utils import get_image_bytes

def refine_box(box, scale, new_width, new_height):
    box = [min(round(box[0]*scale), new_width-1), min(round(box[1]*scale), new_height-1), min(round(box[2]*scale), new_width-1), min(round(box[3]*scale), new_height-1)]
    box = [box[0]/new_width, box[1]/new_height, box[2]/new_width, box[3]/new_height]
    box = [math.floor(x*1000) for x in box]
    if box[0] >= 1000 or box[1] >= 1000 or box[2] >= 1000 or box[3] >= 1000:
        print_rank0(str(box))
        box = [min(box[0], 999), min(box[1], 999), min(box[2], 999), min(box[3], 999)]
    return box

def get_text_by_box(boxlist, sep=" "):
    strs = [f"{box[0]:03d},{box[1]:03d},{box[2]:03d},{box[3]:03d}" for box in boxlist]
    random.shuffle(strs)
    return "{}[[{}]]".format(sep, ";".join(strs))

def parse_resize(img, h, w):
    # if type(img_bytes) is not Image.Image:
    #     img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # else:
    #     img = img_bytes.convert('RGB')
    totalpatch, lpatch = h, w
    # maximize scale s.t.
    scale = math.sqrt(totalpatch * (lpatch / img.size[1]) * (lpatch / img.size[0]))
    num_feasible_rows = max(min(math.floor(scale * img.size[1] / lpatch), totalpatch), 1)
    num_feasible_cols = max(min(math.floor(scale * img.size[0] / lpatch), totalpatch), 1)
    target_height = max(num_feasible_rows * lpatch, 1)
    target_width = max(num_feasible_cols * lpatch, 1)
    # img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
    # img = img.crop((0, 0, target_width, target_height))
    return scale, target_width, target_height

def get_box(b):
    return max(b['x'], 0), max(b['y'], 0), b['x']+b['width'], b['y']+b['height']

def build_sample_input(question, answer, boxes, scale, new_width, new_height):
    new_boxes = [refine_box(box, scale, new_width, new_height) for box in boxes]
    box_txt = [get_text_by_box([box], sep="") for box in new_boxes]
    prompt = f"""{question} Select from:\nA. {box_txt[0]}\nB. {box_txt[1]}\nC. {box_txt[2]}\nD. {box_txt[3]}\n Answer: """
    txt = chr(ord('A')+answer)
    return prompt, txt

def process_data(root_dir, mode):
    data_file = os.path.join(root_dir, 'raw/Visual7W/dataset_v7w_pointing.json')
    save_dir = os.path.join(root_dir, f"processed/Visual7W/{mode}")
    image_dir = os.path.join(root_dir, "raw/Visual7W/images")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(data_file) as f:
        data = json.load(f)
    # box id -> box
    box = data['boxes']
    key2box = {}
    for b in box:
        key2box[b['box_id']] = get_box(b)
    all_data, item_num, drop_num = {}, 0, 0
    for img in data['images']:
        if img['split'] != mode:
            continue
        image_path = os.path.join(image_dir, img["filename"])
        if not os.path.exists(image_path):
            drop_num += 1
            print(f'not found {image_path}.')
            continue
        img_data = Image.open(image_path).convert('RGB')
        scale, new_width, new_height = parse_resize(img_data, 400, 14)
        for qa in img["qa_pairs"]:
            boxes = []
            boxes.append(key2box[qa['answer']])
            for q in qa['multiple_choices']:
                boxes.append(key2box[q])
            permute = [0, 1, 2, 3]
            random.shuffle(permute)
            rand_boxes = [boxes[i] for i in permute]
            answer = permute.index(0)
            prompt, txt = build_sample_input(qa['question'], answer, rand_boxes, scale, new_width, new_height)
            c_data = {
                "question_id": qa["qa_id"],
                "answer": txt,
                "prompt": prompt,
                "type": qa["type"]
            }
            if image_path not in all_data:
                all_data[image_path] = []
            all_data[image_path].append(c_data)
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
            with wds.TarWriter(os.path.join(save_dir, f"{mode}_visual7w_%06d.tar" %(tar_id)), "w") as tar:
                for res in result_tar:
                    tar.write(res)
            result_tar = []
            tar_id += 1
    if len(result_tar) > 0:
        with wds.TarWriter(os.path.join(save_dir, f"{mode}_visual7w_%06d.tar" %(tar_id)), "w") as tar:
            for res in result_tar:
                tar.write(res)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")
    

if __name__ == '__main__':
    root_dir = "/nxchinamobile2/shared/mmbench_datasets"
    for mode in ["test", "val", "train"]:
        print(f"process {mode}.")
        process_data(root_dir, mode)