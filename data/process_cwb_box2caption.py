import os
import json
import random

from utils import save_data

DATASET_NAWE = "CWB_box2caption"

def extract_nouns(origin_txt):
    # 存放匹配到的位置
    positions = []

    start_idx = 0
    while start_idx < len(origin_txt):
        # 查找<ph_st>和<ph_ed>的位置
        start_pos = origin_txt.find("<ph_st>", start_idx)
        end_pos = origin_txt.find("<ph_ed>", start_idx)

        # 如果没有找到更多的匹配项，停止循环
        if start_pos == -1 or end_pos == -1:
            break

        positions.append((start_pos, end_pos))
        start_idx = end_pos + 7  # 加7是因为<ph_ed>有7个字符
    
    return [origin_txt[positions[i][0]+7:positions[i][1]].strip() for i in range(len(positions))]

def process_data(root_dir, mode):
    img_dir = os.path.join(root_dir, f"raw/CWB/flickr30k-images")
    suffix = 'eval' if mode == 'val' else mode
    filename = os.path.join(root_dir, f"raw/CWB/CWB_flickr30k_{suffix}.jsonl")
    save_dir = os.path.join(root_dir, f"processed/CWB_box2caption/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_results = {}
    drop_num, item_num = 0, 0
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp:
            c_data = json.loads(line)
            image_path = os.path.join(img_dir, c_data["image_id"]+".jpg")
            if not os.path.exists(os.path.join(root_dir, image_path)):
                print(f"not found: {image_path}")
                drop_num += 1
                continue
            nouns = extract_nouns(c_data['sentence'])
            for i, (noun, seq) in enumerate(zip(nouns, c_data['boxes_seq'])):
                th_data = {
                    "datatype": "grounding_qa",
                    "question_id": str(c_data["id"]) + str(i),
                    "metadata": {
                        "question": "<ph_st><ph_ed>",
                        "question_boxes": [seq],
                        "answer": noun,
                        "answer_boxes": [],
                        "boxes": c_data["boxes"],
                        "question_type": "box2caption"
                    }
                }
                if image_path not in all_results:
                    all_results[image_path] = []
                all_results[image_path].append(th_data)
                item_num += 1
    # save tarfiles
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")

if __name__ == "__main__":
    root_dir = "/mnt/shared/img_datasets/grounding_stage2"
    for mode in ['train', 'val', 'test']:
        print(f"process {mode}.")
        process_data(root_dir, mode)