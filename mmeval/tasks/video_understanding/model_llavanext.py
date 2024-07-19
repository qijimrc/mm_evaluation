from decord import VideoReader, cpu
import numpy as np
from transformers import AutoConfig
import math
import os
from PIL import Image
import torch
import time

from llavavid.model.builder import load_pretrained_model
from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llavavid.constants import *
from llavavid.conversation import conv_templates, SeparatorStyle


class LLaVANeXT(torch.nn.Module):
    def __init__(self, model_path, conv_mode='llava_llama_2', args=None, device_map='cuda:0'):
        # Set model configuration parameters if they exist
        super(LLaVANeXT, self).__init__()

        self.conv_mode = conv_mode
        model_name = get_model_name_from_path(model_path)
        if args and args.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = args.mm_resampler_type
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_spatial_pool_out_channels"] = args.mm_spatial_pool_out_channels
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["patchify_video_feature"] = False

            cfg_pretrained = AutoConfig.from_pretrained(model_path)

            
            if "224" in cfg_pretrained.mm_vision_tower:
                # suppose the length of text tokens is around 1000, from bo's report
                least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
            else:
                least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

            scaling_factor = math.ceil(least_token_number/4096)
            # import pdb;pdb.set_trace()

            if scaling_factor >= 2:
                if "mistral" not in cfg_pretrained._name_or_path.lower() and "7b" in cfg_pretrained._name_or_path.lower():
                    print(float(scaling_factor))
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device_map=device_map)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len


    def load_video(self, video_path, args):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        # fps = round(vr.get_avg_fps())
        # frame_idx = [i for i in range(0, len(vr), fps)]
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def chat(self,
             msgs,
             tokenizer=None,
             **kwargs,
             ):
        video = msgs[0]['content'][:-1]
        question = msgs[0]['content'][-1]
        device = next(self.parameters()).device

        if isinstance(video, list):
            assert isinstance(video[0], Image.Image) or (isinstance(video[0], np.ndarray) and video[0].ndim==3)
            video = np.stack(video)
        assert video.ndim == 4
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device)
        video = [video]

        model, tokenizer = self.model, self.tokenizer

        qs = question
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        cur_prompt = question
        with torch.inference_mode():
            model.update_prompt([[cur_prompt]])
            # import pdb;pdb.set_trace()
            start_time = time.time()
            import ipdb; ipdb.set_trace()
            output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
            end_time = time.time()
            # print(f"Time taken for inference: {end_time - start_time} seconds")
            # import pdb;pdb.set_trace()
            # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, use_cache=True, stopping_criteria=[stopping_criteria])

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print(f"Question: {prompt}\n")
        # print(f"Response: {outputs}\n")
        # import pdb;pdb.set_trace()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        return outputs


def load_model(model_path, device_id=None, dtype=torch.float32):
    
    # model = LLaVANeXT(model_path, device_map=f'cuda:{device_id}')
    model = LLaVANeXT(model_path)

    if device_id is not None:
        model = model.to(device=f'cuda:{device_id}', dtype=dtype)
        vision_tower = model.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=f'cuda:{device_id}')
        vision_tower.to(device=f"cuda:{device_id}", dtype=dtype)
        # model.model.vision_tower = vision_tower
        torch.cuda.set_device(f"cuda:{device_id}")

    return model, model.tokenizer