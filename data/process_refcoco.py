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
    rec_template = templates['REC']


def build_qa_dataset(instance_f, ann_f, img_dir, split='test', n_samples=-1, save_f='refcoco_sampled.json'):


    with open(instance_f) as f:
        instances = json.load(f)['annotations'] # box with [x, y, w, h] format

    id2insts =  {inst['id']: inst for inst in instances} # this id aligns with the `ann_id` in res.p

    result = []
    tot_succeed  = 0
    with open(ann_f, 'rb') as f:
        annotations = pickle.load(f)
    for ann in tqdm.tqdm(annotations):
        if ann['split'] == split:
            img_path = os.path.join(img_dir, ann['file_name'])
            assert img_path
            inst = id2insts[(ann['ann_id'])]
            bbox = inst['bbox']
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2],  bbox[1]+bbox[3]]
            for sent in ann['sentences']:
                phr = sent['sent']
                
                tmp = random.choice(rec_template)
                if '<image>' in tmp and ' image ' in tmp:
                    tmp = tmp.replace('<image>', '')
                else:
                    tmp = tmp.replace('<image>', 'image')
                
                ex = {
                    'img_path': img_path,
                    'question': tmp.replace('<expr>', phr),
                    'target': bbox,
                    'sentence': phr
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
    parser.add_argument('--data_dir', default='/nxchinamobile2/shared/instruction_data/MultiInstruct/raw/')
    parser.add_argument('--data_names', nargs='+', default=['refcoco', 'refcoco+', 'refcocog'])
    parser.add_argument('--data_anns', nargs='+', default=['refs(google).p', 'refs(unc).p', 'refs(google).p'])
    parser.add_argument('--splits', nargs='+', default=['val', 'val', 'val'])
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
