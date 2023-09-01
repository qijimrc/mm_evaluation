import json
import argparse
from typing import List
import numpy as np
import tqdm
from itertools import combinations
import torch
from transformers import AutoModel, AutoTokenizer




def select_samples(questions_f: List, anns_f: List, N: int=-1, save_f='sampled.json'):
    ''' Select samples to maintain the examples with the most diversified questions, where -1 means select the full set.
    '''
    with open(questions_f) as f1, open(anns_f) as f2:
        questions, anns = json.load(f1), json.load(f2)

    sorted_qs = sorted(questions['questions'], key=lambda x:x['question_id'])
    sorted_anns = sorted(anns['annotations'], key=lambda x:x['question_id'])
    
    if N != -1:
        sampling_ids = np.random.choice(range(len(sorted_qs)), min(N, len(sorted_qs)), replace=False)
    else:
        sampling_ids = range(len(sorted_qs))
    tot_succeed = 0
    qa_annotations = []
    for idx in tqdm.tqdm(sampling_ids):
        q_info, a_info = sorted_qs[idx], sorted_anns[idx]
        qa_annotations.append(q_info | a_info)
        tot_succeed += 1

    questions['annotations'] = qa_annotations
    questions.pop('questions')
    questions['license'] = questions.pop('licence')
    with open(save_f, 'w') as f:
        json.dump(questions, f)
    return tot_succeed, len(sorted_qs)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_file', default='/nxchinamobile2/shared/instruction_data/MultiInstruct/raw/TDIUC/Questions/OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--ann_file', default='/nxchinamobile2/shared/instruction_data/MultiInstruct/raw/TDIUC/Annotations/mscoco_val2014_annotations.json')
    parser.add_argument('--seed', type=int, default=9271)
    parser.add_argument('--N', type=int, default=-1, help='The number of examples for sampling, where -1 means selecting the full set.')
    parser.add_argument('--save_file', default='/nxchinamobile2/shared/instruction_data/evaluation/tdiuc_sampled.jsonl')
    args = parser.parse_args()

    np.random.seed(args.seed)

    tot_select, tot = select_samples(args.question_file, args.ann_file, args.N, args.save_file)
    print(f"Selected {tot_select} samples from total of {tot}")
    