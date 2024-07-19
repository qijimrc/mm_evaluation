import torch
from transformers import AutoModel, AutoTokenizer
import os,sys
import json
import numpy as np
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

from model_llavanext import load_model as load_model_llavanext

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
    model, tokenizer = load_model_llavanext(model_path, device_id=device)

    results = []
    for ex in tqdm.tqdm(vqas, desc=f"[device {device}, rank {rank}]"):
        video_file = ex['video_file']
        question_id = ex['question_id']
        question = ex['question']
        answer = ex.get('answer', None)
        use_dummy = True
        for _ in range(5):
            try:
                vr = VideoReader(str(video_file), ctx=cpu(0))
                sample_fps = round(vr.get_avg_fps() / fps) # FPS
                frame_idx = [i for i in range(0, len(vr), sample_fps)]
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
            })
    if save_f is not None:
        with open(save_f, 'w') as f:
            f.write(json.dumps(results))
    return results



def load_dataset(qa_path, video_dir):
    with open(qa_path) as f:
        data_qa = json.load(f)

    video_dir = video_dir
    # vname2path = {vname.split('.')[0]: os.path.join(video_dir, vname) for vname in os.listdir(video_dir)}
    vqas = [{
        'video_file':os.path.join(video_dir, ex['video_id']),
        'question': ex['question'],
        'question_id': i,
        'answer': ex['answer']
        } for i,ex in enumerate(data_qa)]

    return vqas
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data/qiji/models/llavanext/')
    parser.add_argument('--save_dir', type=str, default='/data/qiji/models/llavanext/output')
    parser.add_argument('--task', type=str, default='tgif_qa')
    parser.add_argument('--num_process', type=int, default=4)
    parser.add_argument('--num_per_device', type=int, default=2)
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--nframes', type=int, default=64)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    print(f"Running for {args.task}")

    vqas = load_dataset(
        qa_path='/data/qiji/video_dataset/tgif/test_qa.json',
        video_dir='/data/qiji/video_dataset/tgif/videos/'
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
                                        kwds={'fps':fps, 'nframes':nframes, "device": i//num_per_device, "save_f": f"{save_dir}/{i}.json", "rank":i})
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

    # from tools.llm_eval_qa import evaluate
    # evaluate(
    #     pred_path=save_result,
    #     output_dir=os.path.join(save_dir, 'chatgpt'),
    #     save_result=os.path.join(save_dir, 'chatgpt_result.json'),
    #     save_metrics=os.path.join(save_dir, 'chatgpt_metrics.json'),
    #     num_tasks=20
    # )

    
