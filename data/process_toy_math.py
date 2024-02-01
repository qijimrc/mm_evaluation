import io
import os
import random
import jsonlines
import pandas as pd
import webdataset as wds

from PIL import Image
from tqdm import tqdm
from utils import get_image_bytes, save_data


from pdb import set_trace as st
import jsonlines

DATASET_NAWE = "ToyMath"

def create_train_test_split(root_dir):
    filename = os.path.join(root_dir, f"meta.jsonl")
    meta_data = []
    with jsonlines.open(filename, "r") as fp:
        for data in fp:
            meta_data.append(data)
    train_ratio = 0.9
    train_num = int(len(meta_data) * train_ratio)
    import random
    random.shuffle(meta_data)
    train_data = meta_data[:train_num]
    test_data = meta_data[train_num:]
    with jsonlines.open(os.path.join(root_dir, f"meta_train.jsonl"), "w") as fp:
        for data in train_data:
            fp.write(data)
    with jsonlines.open(os.path.join(root_dir, f"meta_test.jsonl"), "w") as fp:
        for data in test_data:
            fp.write(data)
            

math_describe = """<|assistant|>
小智识别到这是一条数学题求解意图，题目的 latex 表示为：{latex}
<|assistant|>call_alltools
```python
tool_call(content="请帮我解答下面的数学题：{latex}")
```
<|observation|>
"""

def process_data(root_dir, mode):
    filename = os.path.join(root_dir, f"meta.jsonl")
    img_dir = os.path.join(root_dir, "images")
    save_dir = os.path.join(root_dir, f"../processed/MathToy/")
    os.makedirs(save_dir, exist_ok=True)
    
    # read using jsonlines
    meta_data = []
    with jsonlines.open(filename, "r") as fp:
        for data in fp:
            meta_data.append(data)
    print(f"Total: {len(meta_data)}")
    
    all_results = {}
    drop_num, item_num = 0, 0
    for sub_data in tqdm(meta_data):
        image_path = os.path.join(img_dir, sub_data["img_path"])
        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            drop_num += 1
        if image_path not in all_results:
            all_results[image_path] = []
        c_data = {
            "datatype": "normal_qa",
            "question_id": "%09d" %item_num,
            "metadata": {
                "question": sub_data["question"],
                "answer": math_describe.format(latex=sub_data["answer"])
            }
        }
        all_results[image_path].append(c_data)
        item_num += 1
    # save tarfiles
    all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAWE, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")



if __name__ == "__main__":
    root_dir = "/share/img_datasets/cleaned_instructions/tmp/cya_math_toy/math_toy"
    # create_train_test_split(root_dir)
    # for mode in ["train", "test"]:
        # print(f'process {mode}')
    process_data(root_dir, 'mode')
