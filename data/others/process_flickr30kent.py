from typing import Any, Dict, List
import xml.etree.ElementTree as ET
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

def build_qa_dataset(anns_dir, sents_dir, split_path, img_dir, n_samples=-1, save_f='flickr3kent_sampled.json'):

    with open(split_path) as f:
        split = f.readlines()

    ret_examples = [] # one phrase per sample
    for idx in tqdm.tqdm(split, desc="processing flickr30k-entities .."):
        idx = idx.strip()
        # img
        img_path = os.path.join(img_dir, idx+'.jpg')
        assert os.path.exists(img_path)

        # boxes
        chains = {}
        boxes_ann = os.path.join(anns_dir, idx + '.xml')
        assert os.path.exists(boxes_ann)
        boxes_tree = ET.parse(boxes_ann)
        boxes_root = boxes_tree.getroot()
        for obj in boxes_root.findall("object"):
            if obj.find("./bndbox/xmin") is None:
                continue
            xmin = float(obj.find("./bndbox/xmin").text)
            ymin = float(obj.find("./bndbox/ymin").text)
            xmax = float(obj.find("./bndbox/xmax").text)
            ymax = float(obj.find("./bndbox/ymax").text)
            # draw multiple boxes for all chains
            chain_ids = obj.findall("name")
            for cid in chain_ids:
                cid = cid.text
                if cid not in chains:
                    chains[cid] = [[xmin, ymin, xmax, ymax]]
                else:
                    chains[cid].append([xmin, ymin, xmax, ymax])

        # sentences
        sents_ann = os.path.join(sents_dir, idx + '.txt')
        assert os.path.exists(sents_ann)
        with open(sents_ann) as f: sents = f.readlines()
        for line in sents:
            line = line.strip()
            for piece in re.findall(r'\[.*?\]', line):
                cid = re.match(r'\[/EN#(\d+).*?\]', piece).group(1)
                if cid in chains:
                    phr = re.match(r'\[.*?\s(.*?)\]', piece).group(1)

                    b1, e1 = line.find(piece), line.find(piece)+len(piece)
                    b2, e2 = b1+piece.find(phr), b1+piece.find(phr)+len(phr)

                    tmp = random.choice(rec_template)
                    if '<image>' in tmp and ' image ' in tmp:
                        tmp = tmp.replace('<image>', '')
                    else:
                        tmp = tmp.replace('<image>', 'image')

                    ex = {
                        'img_path': img_path,
                        'question': tmp.replace('<expr>', phr),
                        'target': chains[cid],
                        'sentence': line,
                    }
                    ret_examples.append(ex)

    if n_samples != -1:
        ret_examples = ret_examples[:n_samples]

    if save_f:
        with open(save_f, 'w') as f:
            for ex in ret_examples:
                f.write(json.dumps(ex)+'\n')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns_dir', default='/nxchinamobile2/shared/instruction_data/flickr30k_entities/Annotations')
    parser.add_argument('--sents_dir', default='/nxchinamobile2/shared/instruction_data/flickr30k_entities/Sentences')
    parser.add_argument('--split_path', default='/nxchinamobile2/shared/instruction_data/flickr30k_entities/val.txt')
    parser.add_argument('--img_dir', default='/nxchinamobile2/shared/img_datasets/shikra/flickr30k-images')
    parser.add_argument('--seed', type=int, default=9271)
    parser.add_argument('--N', type=int, default=-1, help='The number of examples for sampling.')
    parser.add_argument('--save_file', default='/nxchinamobile2/shared/instruction_data/evaluation/flickr3kent_sampled.jsonl')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    build_qa_dataset(args.anns_dir, args.sents_dir, args.split_path, args.img_dir, args.N, args.save_file)
    