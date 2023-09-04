# -*- encoding: utf-8 -*-
'''
@File    :   process_okvqa.py
@Time    :   2023/09/01 13:17:50
@Author  :   Jiazheng Xu
@Contact :   xjz22@mails.tsinghua.edu.cn
@Description: Select samples from OKVQA dataset for benchmarking. Here are the points to be considered:
    - Select 500 samples from the standard test set of OKVQA.
    - Select the samples with a maximum diversity of question types.
'''
import json
import argparse
from typing import List
import numpy as np
import tqdm
from itertools import combinations
import torch
from transformers import AutoModel, AutoTokenizer


def select_samples(okvqa_questions_f: List, okvqa_anns_f: List, N: int=500, save_f='okvqa_sampled.json'):
    ''' Select OKVQA samples to maintain the examples with the most diversified questions. (currently peforming random due to the cost)
    '''
    with open(okvqa_questions_f) as f1, open(okvqa_anns_f) as f2:
        questions, anns = json.load(f1), json.load(f2)

    sorted_qs = sorted(questions['questions'], key=lambda x:x['question_id'])
    sorted_anns = sorted(anns['annotations'], key=lambda x:x['question_id'])
    
    sampling_ids = np.random.choice(range(len(sorted_qs)), min(N, len(sorted_qs)), replace=False)
    tot_succeed = 0
    qa_annotations = []
    for idx in tqdm.tqdm(sampling_ids):
        q_info, a_info = sorted_qs[idx], sorted_anns[idx]
        # qa_annotations.append(q_info | a_info)
        qa_annotations.append({**q_info, **a_info})
        tot_succeed += 1

    questions['annotations'] = qa_annotations
    questions.pop('questions')
    with open(save_f, 'w') as f:
        json.dump(questions, f)
    return tot_succeed




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--okvqa_question_file', default='/nxchinamobile2/shared/instruction_data/MultiInstruct/raw/OK-VQA/val/OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--okvqa_ann_file', default='/nxchinamobile2/shared/instruction_data/MultiInstruct/raw/OK-VQA/val/mscoco_val2014_annotations.json')
    parser.add_argument('--seed', type=int, default=9271)
    parser.add_argument('--N', type=int, default=500, help='The number of examples for sampling.')
    parser.add_argument('--save_file', default='/nxchinamobile2/shared/instruction_data/evaluation/okvqa_sampled.json')
    args = parser.parse_args()

    np.random.seed(args.seed)

    select_samples(args.okvqa_question_file, args.okvqa_ann_file, args.N, args.save_file)
    