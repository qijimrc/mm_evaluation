import io
import os
import random
import jsonlines
import pandas as pd
import webdataset as wds

from PIL import Image
from tqdm import tqdm
from utils import save_data



from pdb import set_trace as st
import jsonlines

from data.mmmu_utils import CAT_SHORT2LONG, process_single_sample, construct_prompt
from datasets import concatenate_datasets, load_from_disk


OUR_TEMPLATE = {
    "task_instructions": "",
    "multi_choice_example_format": """{}
    {}
    Short answer.
    """,
    "short_ans_example_format": """{}
    Short answer."
    """,
}

DATASET_NAME = "MMMU"
def process_data(root_dir, split, dataser_sub_name):
    # filename = os.path.join(root_dir, f"meta.jsonl")
    # img_dir = os.path.join(root_dir, "images")
    save_dir = root_dir.replace("datasets", "datasets_processed")
    save_dir = os.path.join(save_dir, split)
    save_images_dir = os.path.join(save_dir, "images")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_images_dir, exist_ok=True)
    
    print(f"Save to {save_dir}")
    # run for each subject
    sub_dataset_list = []
    for subject in tqdm(CAT_SHORT2LONG.values()):
        sub_dir = os.path.join(root_dir, subject, split)
        sub_dataset = load_from_disk(sub_dir)
        sub_dataset_list.append(sub_dataset)
    
    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)
    
    
    all_data = []
    # read using jsonlines
    all_results = {}
    drop_num, item_num = 0, 0
    for sample in tqdm(dataset):
        sample = process_single_sample(sample)
        sample = construct_prompt(sample, OUR_TEMPLATE)

        # save image to save_images_dir
        sample_id = "%09d" %item_num
        image_path = os.path.join(save_images_dir, f"{sample_id}.png")
        image = sample["image"]
        if not os.path.exists(image_path):
            image.save(image_path)
        
        if image_path not in all_results:
            all_results[image_path] = []
        
        orig_dict_without_image = sample.copy()
        del orig_dict_without_image["image"]
        
        c_data = {
            "datatype": "normal_qa",
            "question_id": sample_id,
            "metadata": {
                "question": sample["final_input_prompt"],
                "answer": sample["gt_content"],
                "orig_dict": orig_dict_without_image,
            }
        }
        # print json in a pretty way
        # print(json.dumps(c_data, indent=4))
        
        all_results[image_path].append(c_data)
        item_num += 1
    
    
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, dataser_sub_name, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")
    
    


if __name__ == "__main__":
    root_dir = "/share/home/chengyean/evaluation/cya_ws/datasets/MMMU"

    for mode in ["validation", "test"]:
        
        print(f'process {mode}')
        dataser_sub_name = f"{DATASET_NAME}_{mode}"
        process_data(root_dir, mode, dataser_sub_name)
