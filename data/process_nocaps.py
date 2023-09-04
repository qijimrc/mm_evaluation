from typing import Any, Dict, List
import pickle
import argparse
import tqdm
import os
import json
import re
import random
import numpy as np

with open('templates.json') as f:
    templates = json.load(f)
    cap_template = templates['Caption']


def build_qa_dataset(nocap_ann_f, img_dir, n_samples=-1, save_f='nocap_sampled.json'):


    with open(nocap_ann_f) as f:
        data = json.load(f)
        annotations = data['annotations']
        id2imgs = {info['id']: info for info in data['images']}


    result = []
    tot_succeed  = 0
    for ann in tqdm.tqdm(annotations):
        imginfo = id2imgs[ann['image_id']]
        img_path = os.path.join(img_dir, imginfo['file_name'])
        assert os.path.exists(img_path)

        tmp = random.choice(cap_template)
        if '<image>' in tmp and ' image ' in tmp:
            tmp = tmp.replace('<image>', '')
        else:
            tmp = tmp.replace('<image>', 'image')
        
        ex = {
            'img_path': img_path,
            'question': tmp.replace('<expr>', tmp),
            'target': ann['caption'],
            'image_id': ann['image_id'],
            'open_image_id': imginfo['open_image_id'],
            'domain': imginfo['domain']
        }
        result.append(ex)
        tot_succeed += 1

    tot = len(result)
    if n_samples != -1:
        result = result[:n_samples]

    if save_f:
        with open(save_f, 'w') as f:
            for ex in result:
                f.write(json.dumps(ex)+'\n')
    return tot_succeed, tot

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nocaps_ann', type=str, default="/nxchinamobile2/shared/instruction_data/MultiInstruct/raw/nocaps/nocaps_val_4500_captions.json")
    parser.add_argument('--img_dir', default='/nxchinamobile2/shared/img_datasets/MSCOCO/MSCOCO2014/train2014/')
    parser.add_argument('--seed', type=int, default=9271)
    parser.add_argument('--N', type=int, default=-1, help='The number of examples for sampling.')
    parser.add_argument('--save_dir', default='/nxchinamobile2/shared/instruction_data/evaluation/')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    for dname, dann, split in zip(args.data_names, args.data_anns, args.splits):
        instance_f = os.path.join(args.data_dir, dname, 'instances.json')
        ann_f = os.path.join(args.data_dir, dname, dann)
        save_f = os.path.join(args.save_dir, dname+'_sampled.jsonl')

        tot_succeed, tot = build_qa_dataset(instance_f, ann_f, args.img_dir, split, args.N, save_f)
        print(f"{tot_succeed} example out of {tot} are selected.")
