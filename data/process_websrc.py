import os
import pandas
import random

from utils import save_data

DATASET_NAME = "websrc_html"

def process_data(root_dir, output_dir, mode):
    save_dir = os.path.join(output_dir, f"processed/{DATASET_NAME}/{mode}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if mode in ['train', 'val']:
        all_results = {}
        drop_num, item_num = 0, 0
        split = pandas.read_csv(os.path.join(root_dir, "dataset_split.csv"))
        split_dict = {}
        for index, row in split.iterrows():
            split_dict[(row['domain'], row['website'])] = (row['type'], row['split'] if row['split'] == 'train' else 'val')

        for domain, website in split_dict:
            if split_dict[(domain, int(website))][1] != mode:
                continue
            curr = os.path.join(root_dir, domain, "{:02d}".format(website))
            dataset = pandas.read_csv(os.path.join(curr, "dataset.csv"))
            for index, row in dataset.iterrows():
                question = row['question']
                answer = row['answer']
                _id = row['id']
                answer_start = row['answer_start']
                fn = _id[2:9]
                image_path = os.path.join(curr, "processed_data", fn+".png")
                if not os.path.exists(os.path.join(root_dir, image_path)):
                    print(f"not found: {image_path}")
                    drop_num += 1
                    continue
                if image_path not in all_results:
                    all_results[image_path] = []
                c_data = {
                    "datatype": "normal_qa",
                    "question_id": _id,
                    "metadata": {
                        "question": str(question),
                        "answer": str(answer),
                    }
                }
                all_results[image_path].append(c_data)
                item_num += 1
        # save tarfiles
        all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
        random.shuffle(all_data)
        image_num = save_data(all_data, save_dir, DATASET_NAME, mode)
        print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")
    else:
        all_results = {}
        drop_num, item_num = 0, 0
        root_dir = root_dir+"_testset"
        domains = os.listdir(root_dir)
        for domain in domains:
            websites = os.listdir(os.path.join(root_dir, domain))
            for website in websites:
                curr = os.path.join(root_dir, domain, website)
                dataset = pandas.read_csv(os.path.join(curr, "dataset.csv"))
                for index, row in dataset.iterrows():
                    question = row['question']
                    # answer = row['answer']
                    _id = row['id']
                    # answer_start = row['answer_start']
                    fn = _id[2:9]
                    image_path = os.path.join(curr, "processed_data", fn+".png")
                    html_path = os.path.join(curr, "processed_data", fn+".html")
                    with open(html_path, "r", encoding="utf-8") as f:
                        html = f.read()
                    if not os.path.exists(os.path.join(root_dir, image_path)):
                        print(f"not found: {image_path}")
                        drop_num += 1
                        continue
                    if image_path not in all_results:
                        all_results[image_path] = []
                    c_data = {
                        "datatype": "html_qa",
                        "question_id": _id,
                        "metadata": {
                            "question": str(question),
                            "answer": "",
                            "html": html
                        }
                    }
                    all_results[image_path].append(c_data)
                    item_num += 1
        # save tarfiles
        all_data = [{"image_path": key, "json": value} for key, value in all_results.items()]
        random.shuffle(all_data)
        image_num = save_data(all_data, save_dir, DATASET_NAME, mode)
        print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_num} samples")


if __name__ == '__main__':
    root_dir = "/share/home/lqs/benchmark/WebSRC/release"
    output_dir = "/share/home/lqs/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    for mode in ['train', 'val', 'test']:
        print(f"process {mode}.")
        process_data(root_dir, output_dir, mode)