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
    # torch.cuda.set_device(f"cuda:0")
    model, tokenizer = load_model_llavanext(model_path, device_id=device)

    results = []
    for ex in tqdm.tqdm(vqas, desc=f"[device {device}, rank {rank}]"):
        video_file = ex['video_file']
        question_id = ex['question_id']
        question_list = ex['question'] # A LIST OF QUESTIONS (1 or 2)
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

        # A list of questions
        res_list = []
        for question in question_list:
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
            res_list.append(res)
        # print(f"Question: {question}\t Answer: {gt}\t Pred: {res}")
        if len(question_list) == 1:
            results.append({'pred': res_list[0], "question_id": question_id, 'question':question, 'answer': answer})
        else:
            results.append({'pred1': res_list[0], 'pred2': res_list[1], "question_id": question_id, 
                            'question1':question_list[0],'question2':question_list[1], 'answer': answer})

    if save_f is not None:
        with open(save_f, 'w') as f:
            f.write(json.dumps(results))
    return results

    
def calculate_metrics(
        res_dict_list: List[Dict],
        save_f: str=None) -> Dict:
    """
    Params:
      @res_dict_list: [{'question', 'question_id', 'pred', 'truth'}]
    """
    tot, tot_yn = 0, 0
    tot_correct, tot_yn_correct = 0, 0
    for ex in res_dict_list:
        truth = ex['answer']
        pred = ex['pred']
        if 'sorry' in pred:
            continue
        tot += 1
        yn_question = False
        if truth.strip().translate(str.maketrans('', '', string.punctuation)).lower() in ['yes', 'no']:
            yn_question = True
            tot_yn += 1
        
        if pred.strip().lower() in truth.strip().lower() or \
            truth.strip().lower() in pred.strip().lower()[:10].translate(str.maketrans('', '', string.punctuation)):
            tot_correct += 1
            if yn_question:
                tot_yn_correct += 1

    acc = tot_correct / tot
    yn_acc = tot_yn_correct / tot_yn
    metrics = {'Tot': tot, 'Acc': acc, 'YesNoAcc':yn_acc}
    print(str(metrics))
    if save_f is not None:
        with open(save_f, 'w') as f:
            f.write(json.dumps(metrics))


def load_dataset(qa_path, video_dir):
    # ActivityNet-QA
    with open(qa_path) as f:
        data_q = json.load(f)
    
    video_dir = video_dir
    vname2path = {vname.split('.')[0]: os.path.join(video_dir, vname) for vname in os.listdir(video_dir)}
    vqas = [{
        'video_file':vname2path[ex['video_name']],
        'question': [ex['Q']] if 'Q' in ex else [ex['Q1'], ex['Q2']], # a list
        'question_id': i,
        'answer': ex['A']
        } for i, ex in enumerate(data_q)]

    return vqas

class generic_qa:
    prompts = {
        'correctness_prompts' : {'system': "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                            "- The predicted answer must be factually accurate and align with the video content.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the factual accuracy of the prediction compared to the answer.",
                'input_text': "Please evaluate the following video-based question-answer pair:\n\n"
                                "Question: {question}\n"
                                "Correct Answer: {answer}\n"
                                "Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {{''score': 4.8}}."
        },

        'detailed_orientation_prompts' : {'system':  "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                                                "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
                                                "------"
                                                "##INSTRUCTIONS: "
                                                "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                                                "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                                                "- Consider synonyms or paraphrases as valid matches.\n"
                                                "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.",
                'input_text': "Please evaluate the following video-based question-answer pair:\n\n"
                                "Question: {question}\n"
                                "Correct Answer: {answer}\n"
                                "Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {{''score': 4.8}}."
        },

        'context_prompts': {'system':   "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                            "- The predicted answer must capture the main themes and sentiments of the video.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Provide your evaluation of the contextual understanding of the prediction compared to the answer.",
                'input_text': "Please evaluate the following video-based question-answer pair:\n\n"
                                "Question: {question}\n"
                                "Correct Answer: {answer}\n"
                                "Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {{''score': 4.8}}."
        }
    }

    def calc_scores(self, combined_contents):
        score_sum = 0
        count = 0
        for key, result in combined_contents.items():
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        average_score = score_sum / count
        metrics = {'average_score': average_score}
        print(str(metrics))
        return metrics



class temporal_qa:
    prompts = {'temporal_prompts' : {'system': "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
                            "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
                            "- Evaluate the temporal accuracy of the prediction compared to the answer.",
                'input_text': "Please evaluate the following video-based question-answer pair:\n\n"
                            "Question: {question}\n"
                            "Correct Answer: {answer}\n"
                            "Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of temporal consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {{''score': 4.8}}."
        }}


    def calc_scores(self,combined_contents):
        score_sum = 0
        count = 0
        for key, result in combined_contents.items():
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        average_score = score_sum / count
        metrics = {'average_score': average_score}
        print(str(metrics))
        return metrics

class consistency_qa:
    prompts = {'consistency_prompts' : {'system': "You are an intelligent chatbot designed for evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
                            "You will be given two very similar questions, a common answer common to both the questions and predicted answers for the two questions ."
                            "Your task is to compare the predicted answers for two very similar question, with a common correct answer and determine if they are consistent. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or significant differences in the conveyed information.\n"
                            "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
                            "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
                            "- Evaluate the consistency of the two predicted answers compared to the correct answer.",
                'input_text': "Please evaluate the following video-based question-answer pair:\n\n"
                            "Question 1: {question1}\n"
                            "Question 2: {question2}\n"
                            "Correct Answer: {answer}\n"
                            "Predicted Answer to Question 1: {pred1}\n"
                            "Predicted Answer to Question 2: {pred2}\n\n"
                            "Provide your evaluation only as a consistency score where the consistency score is an integer value between 0 and 5, with 5 indicating the highest level of consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the consistency score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {{''score': 4.8}}."
        }}


    def calc_scores(self,combined_contents):
        score_sum = 0
        count = 0
        for key, result in combined_contents.items():
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        average_score = score_sum / count
        metrics = {'average_score': average_score}
        print(str(metrics))
        return metrics
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data/qiji/models/llavanext/')
    parser.add_argument('--save_dir', type=str, default='/data/qiji/models/llavanext/output')
    parser.add_argument('--task', type=str, default='videochatgpt_qa')
    parser.add_argument('--num_process', type=int, default=4)
    parser.add_argument('--num_per_device', type=int, default=2)
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--nframes', type=int, default=64)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    for subset in ['generic_qa', 'temporal_qa','consistency_qa']:

        vqas = load_dataset(
            qa_path=f'/data/qiji/video_dataset/videoChat_{subset}.json',
            video_dir='/data/qiji/video_dataset/activity/videos'
        )
        
        prompt = "Please give helpful, detailed, and polite answers to the human's questions."

        # load model
        model_path = args.model_path
        save_dir = os.path.join(args.save_dir, args.task, subset)
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

        EQA = eval(f'{subset}()')
        for subsubset in EQA.prompts:
            system = EQA.prompts[subsubset]['system']
            input_text = EQA.prompts[subsubset]['input_text']
            calc_scores = EQA.calc_scores
            from tools.llm_eval_qa import evaluate
            evaluate(
                pred_path=save_result,
                output_dir=os.path.join(save_dir, 'chatgpt', subsubset),
                save_result=os.path.join(save_dir, f'chatgpt_{subsubset}_result.json'),
                save_metrics=os.path.join(save_dir, f'chatgpt_{subsubset}_metrics.json'),
                num_tasks=20,
                system=system,
                input_text=input_text,
                calc_scores=calc_scores
            )

    

