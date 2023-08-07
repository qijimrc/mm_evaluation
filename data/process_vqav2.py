# -*- encoding: utf-8 -*-
'''
@File    :   process_vqav2.py
@Time    :   2023/08/05 20:36:50
@Author  :   Ji Qi
@Contact :   qij20@mails.tsinghua.edu.cn
@Description: Select samples from VQAv2 dataset for benchmarking. Here are the points to be considered:
    - Select 500 samples from the standard test set of VQAv2.
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


def calc_sentences_similarities(model: AutoModel, tokenizer: AutoTokenizer, sentences:List):
    ''' Calculate the cosine similarities between all pairs of sentences based on the given model.
    '''
    similarities, indices = [], []
    for x, y in combinations(range(len(sentences)), 2):
        feat_x = model(tokenizer(sentences[x]))
        feat_y = model(tokenizer(sentences[y]))
        sim = torch.cosine_similarity(torch.flatten(feat_x), torch.flatten(feat_y))
        similarities.append(sim)
        indices.append([x, y])
    return similarities, indices



def select_samples(vqav2_questions_f: List, vqav2_anns_f: List, N: int=500, save_f='vqav2_sampled.json'):
    ''' Select VQAv2 samples to maintain the examples with the most diversified questions. (currently peforming random due to the cost)
    '''
    with open(vqav2_questions_f) as f1, open(vqav2_anns_f) as f2:
        questions, anns = json.load(f1), json.load(f2)

    sorted_qs = sorted(questions['questions'], key=lambda x:x['question_id'])
    sorted_anns = sorted(anns['annotations'], key=lambda x:x['question_id'])
    
    sampling_ids = np.random.choice(range(len(sorted_qs)), min(N, len(sorted_qs)), replace=False)
    tot_succeed = 0
    qa_annotations = []
    for idx in tqdm.tqdm(sampling_ids):
        q_info, a_info = sorted_qs[idx], sorted_anns[idx]
        qa_annotations.append(q_info | a_info)
        tot_succeed += 1

    questions['annotations'] = qa_annotations
    questions.pop('questions')
    with open(save_f, 'w') as f:
        json.dump(questions, f)
    return tot_succeed




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqav2_question_file', default='/nxchinamobile2/shared/instruction_data/MultiInstruct/raw/VQA_V2/v2_OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--vqav2_ann_file', default='/nxchinamobile2/shared/instruction_data/MultiInstruct/raw/VQA_V2/v2_mscoco_val2014_annotations.json')
    parser.add_argument('--seed', type=int, default=9271)
    parser.add_argument('--N', type=int, default=500, help='The number of examples for sampling.')
    parser.add_argument('--save_file', default='/nxchinamobile2/shared/instruction_data/evaluation/vqav2_sampled.jsonl')
    args = parser.parse_args()

    np.random.seed(args.seed)

    select_samples(args.vqav2_question_file, args.vqav2_ann_file, args.N, args.save_file)
    