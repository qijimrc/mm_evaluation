import torch
from transformers import AutoModel, AutoTokenizer
import os,sys
import json
import numpy as np
from collections import defaultdict
import string
import math
import tqdm
import itertools
import multiprocessing
from PIL import Image
from argparse import ArgumentParser
from decord import VideoReader, cpu
from pathlib import Path
from typing import Dict, List
import argparse
import pyarrow.parquet as pq
from functools import partial

from model_llavanext import load_model as load_model_llavanext

def read_frame(frames_path, num_frames, bound, fps=3):
        max_frame = len(os.listdir(frames_path))
        images_group = list()
        # get indices
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(1, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_frames
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_frames)
        ])
        
        for frame_index in frame_indices:
            img_path = os.path.join(frames_path, f"{frame_index:05d}.jpg")
            if not os.path.exists(img_path):
                print(f"Frame not found for {img_path} So ignoring !!!")
                continue
            img = Image.open(img_path)
            images_group.append(np.array(img))
        video = np.stack(images_group)
        return video

def run(
        model_path: str,
        vqas: List[Dict],
        prompt: str=None,
        device: int=0,
        fps: int=1,
        nframes: int=64,
        save_f: str=None,
        rank: int=0) -> List:
    """
    Params:
        @vqas: [{'video_path', 'question_id', 'question', 'answer[optional]', ...}, ...]
    """
    # load model
    # torch.cuda.set_device(f"cuda:0")
    model, tokenizer = load_model_llavanext(model_path, device_id=device)

    results = []
    for ex in tqdm.tqdm(vqas, desc=f"[device {device}, rank {rank}]"):
        task_type = ex['task_type']
        video_file = ex['video_file']
        question_id = ex['question_id']
        question = ex['question']
        answer = ex.get('answer', None)
        bound = ex.get('bound', False)
        data = ex.get('data', None)
        data_type = ex['data_type']
        use_dummy = True
        for _ in range(5):
            try:
                if data_type == 'frame':
                    video = read_frame(video_file, nframes, (data['start'], data['end']) if bound else None)
                    break
                vr = VideoReader(str(video_file), ctx=cpu(0), num_threads=1)
                start_idx, end_idx = 0, len(vr)
                if bound:
                    ori_fps = vr.get_avg_fps()
                    start, end = data['start'], data['end']
                    start_idx = max(0, round(start * ori_fps))
                    end_idx = min(round(end * ori_fps), len(vr))
                sample_fps = round(vr.get_avg_fps() / fps) # FPS
                frame_idx = [i for i in range(start_idx, end_idx, sample_fps)]
                frame_idx = [frame_idx[i] for i in np.linspace(0, len(frame_idx)-1, min(nframes, len(frame_idx)), dtype=int)]
                # video = vr.get_batch(frame_idx).asnumpy()
                video = vr.get_batch(frame_idx).asnumpy()
                use_dummy = False
                break
            except BaseException as e:
                video = np.zeros([nframes, 448,448,3], dtype=np.uint8)
        
        if use_dummy:
            print(f"Using dummy video as loading failed for {video_file}!!!")
        # t, h, w, c = video.shape
        # video = Image.fromarray(video.reshape(t*h, w, c)).convert('RGB').resize([448, 448*t])
        video = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in video]
        # image = torch.cat(torch.split(video, 1, dim=0), 1).squeeze(0)

        # msgs = [{'role': 'user', 'content': question + " Please answer me yes or no."}]
        question = question + '\n' + prompt if prompt is not None else question
        inp_msgs = [{'role': 'user', 'content': video + [question]}]

        assert video is not None
        sampling = True
        res = model.chat(
                image=None,
                msgs=inp_msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=sampling,
                num_beams=3,
                max_new_tokens=100,
                temperature=0.2,
                max_inp_length=2048 * 10,
            )
        # print(f"Question: {question}\t Answer: {gt}\t Pred: {res}")
        results.append({
            'pred': res,
            "question_id": question_id,
            'question':question,
            'answer': answer,
            'video_id': ex.get('video_id', None),
            'duration': ex.get('duration', None),
            'task_type': task_type,
            })
    if save_f is not None:
        with open(save_f, 'w') as f:
            f.write(json.dumps(results))
    return results



def load_dataset(json_path, video_dir):

    def qa_template(data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        # answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        answer = f"({chr(ord('A') + answer_idx)})"
        return question, answer
    

    data_list = {
        "Action Sequence": ("action_sequence.json", f"{video_dir}/star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", f"{video_dir}/star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("action_antonym.json", f"{video_dir}/ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", f"{video_dir}/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", f"{video_dir}/FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", f"{video_dir}/clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", f"{video_dir}/star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", f"{video_dir}/perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", f"{video_dir}/clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", f"{video_dir}/sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", f"{video_dir}/scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", f"{video_dir}/perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", f"{video_dir}/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", f"{video_dir}/clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", f"{video_dir}/perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", f"{video_dir}/nturgbd/", "video", False),
        "Character Order": ("character_order.json", f"{video_dir}/perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", f"{video_dir}/vlnqa/", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", f"{video_dir}/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", f"{video_dir}/clevrer/video_validation/", "video", False),
    }

    vqas = []
    tot = 0
    for k, v in data_list.items():
        with open(os.path.join(json_path, v[0]), 'r') as f:
            json_data = json.load(f)
        for data in json_data:
            # get question, answer
            question, answer = qa_template(data)
            video_path = os.path.join(v[1], data['video'])
            vqas.append({
                'task_type': k,
                'prefix': v[1],
                'data_type': v[2],
                'bound': v[3],
                'question_id': tot,
                'question': question,
                'answer': answer,
                'video_file': video_path,
                'data': data,
            })
            tot += 1

    return vqas
    
def calculate_metrics(
        res_dict_list: List[Dict],
        save_f: str=None) -> Dict:
    """
    Params:
      @res_dict_list: [{'question', 'question_id', 'pred', 'truth'}]
    """
    # tot_correct, tot = 0, 0
    tot_correct, tot = defaultdict(int), defaultdict(int) # per task-type
    for ex in res_dict_list:
        task_type = ex['task_type']
        truth = ex['answer']
        pred = ex['pred']
        if 'sorry' in pred:
            continue
        truth = truth.strip()
        if truth.startswith('(') and truth.endswith(')'):
            truth = truth[1:-1]
        # tot += 1
        tot[task_type] += 1

        if pred.strip().lower() in truth.strip().lower() or \
            truth.strip().lower() in pred.strip().lower()[:10].translate(str.maketrans('', '', string.punctuation)):
            # tot_correct += 1
            tot_correct[task_type] += 1

    metrics = {}
    sum_tc, sum_t = 0., 0.
    for task_type in tot:
        acc = tot_correct[task_type] / tot[task_type]
        metrics[metrics] = acc
        sum_tc, sum_t = sum_tc+tot_correct[task_type], sum_t+tot[task_type]
    metrics['avg_acc'] = sum_tc / sum_t
    print(str(metrics))
    if save_f is not None:
        with open(save_f, 'w') as f:
            f.write(json.dumps(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data/qiji/models/llavanext/')
    parser.add_argument('--save_dir', type=str, default='/data/qiji/models/llavanext/output')
    parser.add_argument('--task', type=str, default='mvbench')
    parser.add_argument('--num_process', type=int, default=4)
    parser.add_argument('--num_per_device', type=int, default=2)
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--nframes', type=int, default=64)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    print(f"Running for {args.task}")

    prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question."

    vqas = load_dataset(
        json_path='/data/qiji/video_dataset/MVBench/json',
        video_dir='/data/qiji/video_dataset/MVBench/video'
    )

    # load model
    model_path = args.model_path
    save_dir = os.path.join(args.save_dir, args.task)
    os.makedirs(save_dir, exist_ok=True)
    num_process = min(args.num_process, len(vqas))
    fps = args.fps
    nframes = args.nframes
    num_per_device = args.num_per_device
    save_result = os.path.join(save_dir, f'results_{args.task}_fps={args.fps}_nframs={args.nframes}.json')
    save_metrics = os.path.join(save_dir, f'metrics_{args.task}_fps={args.fps}_nframs={args.nframes}.json')

    # Run on multiple processes
    if not os.path.exists(save_result):
        if num_process > 1:
            chunk_size = len(vqas) // num_process + int(bool(len(vqas) % num_process))
            chunk_src = [vqas[i: i+chunk_size] for i in range(0, len(vqas), chunk_size)]
            pool = multiprocessing.Pool(processes=num_process)
            tot = 0
            results = []
            for i in range(len(chunk_src)):
                results.append(
                        pool.apply_async(run,
                                        args=(model_path, chunk_src[i],),
                                        kwds={'prompt': prompt, 'fps':fps, 'nframes':nframes, "device": i//num_per_device, "save_f": f"{save_dir}/{i}.json", "rank":i})
                    )
                # results = run(model_path, chunk_src[i], fps=fps, nframes=nframes, device=0, save_f= f"{save_dir}/{i}.json")
            pool.close(); pool.join()
            results = list(itertools.chain(*[rt.get() for rt in results]))
        else:
            results = run(model_path, vqas, fps=fps, nframes=nframes, device=0, save_f= f"0.json")

        # save
        with open(save_result, 'w') as f:
            f.write(json.dumps(results))
        
    # Calculate metrics
    with open(save_result) as f:
        results = json.load(f)
    calculate_metrics(results, save_metrics)

    
