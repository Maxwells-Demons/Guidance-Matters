import json
import numpy as np
import math
import csv
import random
import argparse
import torch
import os
import torch.distributed as dist
import torch.nn.functional as F


# from copy import deepcopy
from accelerate.utils import set_seed
from diffusers.utils.torch_utils import randn_tensor
from utils import *


device = torch.device('cuda')

def pipe_handler(model, method):
    if model == 'sdxl':
        if method == 'zigzag':
            from pipelines.sdxl_zigzag_1 import StableDiffusionXLPipeline
            from diffusers import DDIMScheduler, DDIMInverseScheduler
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                                    subfolder='scheduler')
            pipe.inv_scheduler = inverse_scheduler

            pipe.forward_optim = pipe.optim_call_way_6
            nfe_ratio = 3
        elif method == 'cfgpp':
            from pipelines.sdxl_cfgpp import StableDiffusionXLPipeline, EulerDiscreteScheduler
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16
            )

            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.forward_cfgpp
            nfe_ratio = 1
        elif method == 'pag':
            from pipelines.sdxl_pag import StableDiffusionXLPAGPipeline
            from diffusers import EulerDiscreteScheduler
            pipe = StableDiffusionXLPAGPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16,
                pag_applied_layers='mid'
            )

            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1.5
        elif method == 'w2sd':
            from pipelines.sdxl_w2sd import StableDiffusionXLPipeline
            from diffusers import DDIMScheduler, DDIMInverseScheduler
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16,
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.inv_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                                    subfolder='scheduler')
            lora_name = 'user_lora'
            pipe.load_lora_weights(args.lora_path, adapter_name=lora_name)
            pipe.forward_optim = pipe.w2sd_lora
            nfe_ratio = 3
        elif method == 'apg':
            from pipelines.sdxl_apg import StableDiffusionXLPipeline
            from diffusers import EulerDiscreteScheduler
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16,
            )

            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1
        elif method == 'freeu':
            from pipelines.sdxl_freeu import StableDiffusionXLPipeline
            from diffusers import EulerDiscreteScheduler
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16,
            )

            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1
        elif method == 'tdg':
            from pipelines.sdxl_tdg import StableDiffusionXLPipeline
            from diffusers import EulerDiscreteScheduler
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16,
            )
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.forward_v3
            nfe_ratio = 1.5
        elif method == 'sag':
            from pipelines.sdxl_sag import StableDiffusionXLPipeline
            from diffusers import DDIMScheduler
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16,
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1.5
        elif method == 'seg':
            from pipelines.sdxl_seg import StableDiffusionXLSEGPipeline
            pipe = StableDiffusionXLSEGPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16,
            )
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1.5
        else:
            raise NotImplementedError
    
    elif model == 'sd35':
        from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            "/data/shared_data/SDV35/sdv35/",
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )
        if method == 'zigzag':
            from pipelines.sd35_zigzag import StableDiffusion3Pipeline, FlowMatchEulerInverseScheduler
            from diffusers import FlowMatchEulerDiscreteScheduler
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "/data/shared_data/SDV35/sdv35/", 
                transformer=model_nf4,
                torch_dtype=torch.bfloat16
            )
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            inverse_scheduler = FlowMatchEulerInverseScheduler.from_pretrained("/data/shared_data/SDV35/sdv35/",
                                                                    subfolder='scheduler')
            pipe.inv_scheduler = inverse_scheduler
            pipe.forward_optim = pipe.forward_v3
            nfe_ratio = 3
        elif method == 'cfgpp':
            from pipelines.sd35_cfgpp import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "/data/shared_data/SDV35/sdv35/", 
                transformer=model_nf4,
                torch_dtype=torch.bfloat16
            )

            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.forward_cfgpp
            nfe_ratio = 1
        elif method == 'pag':
            from pipelines.sd35_pag import StableDiffusion3PAGPipeline
            from diffusers import FlowMatchEulerDiscreteScheduler
            pipe = StableDiffusion3PAGPipeline.from_pretrained(
                "/data/shared_data/SDV35/sdv35/", 
                transformer=model_nf4,
                torch_dtype=torch.bfloat16,
                enable_pag=True,
                pag_applied_layers=["blocks.13"]
            )

            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1.5
        elif method == 'apg':
            from pipelines.sd35_apg import StableDiffusion3Pipeline
            from diffusers import FlowMatchEulerDiscreteScheduler
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "/data/shared_data/SDV35/sdv35/", 
                transformer=model_nf4,
                torch_dtype=torch.bfloat16,
            )

            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1
        elif method == 'freeu':
            raise NotImplementedError
            from pipelines.sd35_freeu import StableDiffusion3Pipeline
            from diffusers import FlowMatchEulerDiscreteScheduler
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "/data/shared_data/SDV35/sdv35/", 
                transformer=model_nf4,
                torch_dtype=torch.bfloat16,
            )

            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1
        elif method == 'tdg':
            from pipelines.sd35_tdg import StableDiffusion3Pipeline
            from diffusers import FlowMatchEulerDiscreteScheduler
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "/data/shared_data/SDV35/sdv35/", 
                transformer=model_nf4,
                torch_dtype=torch.bfloat16,
            )

            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.forward_v3
            nfe_ratio = 1.5
        else:
            raise NotImplementedError
    elif model == 'sd21':
        if method == 'zigzag':
            from pipelines.sd_zigzag import StableDiffusionPipeline
            from diffusers import DDIMScheduler, DDIMInverseScheduler
            pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            inverse_scheduler = DDIMInverseScheduler.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                                    subfolder='scheduler')
            pipe.inv_scheduler = inverse_scheduler

            nfe_ratio = 3
        elif method == 'cfgpp':
            from pipelines.sd_cfgpp import StableDiffusionPipeline, EulerDiscreteScheduler
            # from diffusers import EulerDiscreteScheduler
            pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16
            )
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            nfe_ratio = 1
            # pipe.forward_optim = pipe.__call__
        elif method == 'pag':
            from pipelines.sd_pag import StableDiffusionPAGPipeline
            from diffusers import EulerDiscreteScheduler
            pipe = StableDiffusionPAGPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16,
                pag_applied_layers=['mid']
            )
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            nfe_ratio = 1.5
            pipe.forward_optim = pipe.__call__
        elif method == 'apg':
            from pipelines.sd_apg import StableDiffusionPipeline
            from diffusers import EulerDiscreteScheduler
            pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16,
            )
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1
        elif method == 'freeu':
            from pipelines.sd_freeu import StableDiffusionPipeline
            from diffusers import EulerDiscreteScheduler
            pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16,
            )
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1
        elif method == 'sag':
            from pipelines.sd_sag import StableDiffusionSAGPipeline
            from diffusers import DDIMScheduler
            pipe = StableDiffusionSAGPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1", 
                use_safetensors=True, variant="fp16",
                torch_dtype=torch.float16
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.forward_optim = pipe.__call__
            nfe_ratio = 1.5
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    return pipe, nfe_ratio

def get_args():
    # pick: test_unique_caption_zh.csv       draw: drawbench.csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='sdxl', choices=['sdxl', 'sd35', 'sd21'], type=str)
    parser.add_argument("--method", default='zigzag', type=str)
    parser.add_argument("--inference_step", default=50, type=int)
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--T_max", default=1, type=int)
    parser.add_argument("--seed", default=1000, type=int)
    parser.add_argument("--RatioT", default=1.0, type=float)    #if RatioT==1,则退化为start point优化
    parser.add_argument("--denoising_cfg", default=5.5, type=float)
    parser.add_argument("--inversion_cfg", default=0, type=float)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=None, type=int)
    parser.add_argument("--prompt_path", default='datasets/test_unique_caption_zh.csv', type=str)
    parser.add_argument("--dataset", default='pick', type=str)

    parser.add_argument("--lora_sclae", type=float, default=0.8)
    parser.add_argument("--lora_path", type=str, default='./ckpt/xlMoreArtFullV1.pREw.safetensors')

    parser.add_argument("--weak_lora_scale", type=float, default=-1.5)
    parser.add_argument("--strong_lora_scale", type=float, default=1.5)

    parser.add_argument("--weak_guidance_scale", type=float, default=1.0)
    parser.add_argument("--strong_guidance_scale", type=float, default=5.5)
    
    args =  parser.parse_args()
    return args

if __name__ == '__main__':
    torch.cuda.empty_cache()
    dtype = torch.float16
    args = get_args()
    print("args.seed: ", args.seed)
    set_seed(args.seed)

    if args.dataset is not None:
        base_dir = f'./results/{args.dataset}_{args.method}/output_{args.model}_train_{args.seed}/'
    else:
        base_dir = f'./results/{os.path.basename(args.prompt_path)}_{args.method}/output_{args.model}_train_{args.seed}/'

    

    if args.dataset == 'pick':
        args.prompt_path = 'datasets/test_unique_caption_zh.csv'
        prompt_list, seed_list = load_prompt(path=args.prompt_path, prompt_version='pick')
    elif args.dataset == 'draw':
        args.prompt_path = 'datasets/drawbench.csv'
        prompt_list, seed_list = load_prompt(path=args.prompt_path, prompt_version='draw')
    elif args.dataset == 'hpd':
        args.prompt_path = 'datasets/HPD/all.txt'
        prompt_list, seed_list = load_prompt(path=args.prompt_path, prompt_version='hpsv2')
    elif args.dataset == 'coco':
        prompt_list = load_coco_prompt(length=200)
    else:
        prompt_list, seed_list = load_prompt(path=args.prompt_path)
    
    if args.end_idx is None:
        args.end_idx = len(prompt_list)
    finished = True
    if args.start_idx is not None and args.end_idx is not None:
        for idx in range(args.start_idx, args.end_idx):
            if os.path.exists(os.path.join(base_dir, 'json', 'new{:07d}.json'.format(idx))):
                continue
            else:
                finished = False
                break
    if finished:
        print(f'({args.start_idx}, {args.end_idx}) finished.')
        exit()

    pipe, nfe_ratio = pipe_handler(args.model, args.method)
    pipe.to(device)
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    if not os.path.exists(os.path.join(base_dir,'optim')):
        os.makedirs(os.path.join(base_dir,'optim'), exist_ok=True)
    if not os.path.exists(os.path.join(base_dir,'json')):
        os.makedirs(os.path.join(base_dir,'json'), exist_ok=True)
    if not os.path.exists(os.path.join(base_dir,'latents')):
        os.makedirs(os.path.join(base_dir,'latents'), exist_ok=True)

    evaluator = Evaluator()
    
    
    size = args.size
    if args.model == 'sdxl' or args.model == 'sd21':
        shape = (1, 4, size // 8, size // 8)
    elif args.model == 'sd35':
        shape = (1, 16, size // 8, size // 8)
        
    num_steps = args.inference_step
    guidance_scale = args.denoising_cfg
    inversion_guidance_scale = args.inversion_cfg

    # if args.method == 'cfgpp':
    #     guidance_scale = 0.4
    pag_scale = 3
    lora_name = 'user_lora'
    # if args.method == 'freeu':
    #     pipe.unet2 = deepcopy(pipe.unet)
    b1=1.2
    b2=1.4
    s1=0.9
    s2=0.2


    WINNING_NUMBER = 0.0
    OPTIM_SCORE = 0.0
    ORIGINAL_SCORE = 0.0
    
    _ = 0
    idx_list = list(range(args.start_idx, args.end_idx))
    # random.shuffle(idx_list)
    for idx in idx_list:
        prompt = prompt_list[idx]
        print('idx:', idx)
        print('prompt:', prompt)
        # print(os.path.join(base_dir, 'json', 'new{:07d}.json'.format(idx)))
        if os.path.exists(os.path.join(base_dir, 'json', 'new{:07d}.json'.format(idx))):
            print("pass this prompt (existed): ", prompt)
            continue
        if prompt == '':
            continue
        start_latents = randn_tensor(shape, dtype=dtype, device=device, generator=torch.Generator('cuda').manual_seed(args.seed+idx))
        
        # negative_prompt="worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark",
        optim_kwargs = dict()
        if args.method == 'zigzag':
            optim_kwargs['inv_cfg'] = [inversion_guidance_scale]
            optim_kwargs['T_max'] = 1
        elif args.method == 'cfgpp':
            if args.model == 'sdxl' or args.model == 'sd21':
                guidance_scale = 0.4
            elif args.model == 'sd35':
                guidance_scale = 0.1
        elif args.method == 'pag':
            optim_kwargs['pag_scale'] = pag_scale
        elif args.method == 'w2sd':
            pipe.disable_lora()
            optim_kwargs['denoise_lora_scale'] = args.lora_sclae
            optim_kwargs['lora_gap_list'] = [args.strong_lora_scale,args.weak_lora_scale]
            optim_kwargs['cfg_gap_list'] = [args.strong_guidance_scale,args.weak_guidance_scale]
            optim_kwargs['lora_name'] = lora_name
        elif args.method == 'apg':
            optim_kwargs['adaptive_projected_guidance'] = True
        elif args.method == 'freeu':
            # register_free_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2) # type: ignore
            # register_free_crossattn_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2) # type: ignore
            optim_kwargs['do_freeu'] = True
            if args.model == 'sdxl':
                optim_kwargs['b1']=1.3
                optim_kwargs['b2']=1.4
                optim_kwargs['s1']=0.9
                optim_kwargs['s2']=0.2
            elif args.model == 'sd21':
                optim_kwargs['b1']=1.4
                optim_kwargs['b2']=1.6
                optim_kwargs['s1']=0.9
                optim_kwargs['s2']=0.2
        elif args.method == 'tdg':
            if args.model == 'sd35':
                optim_kwargs['guidance_scale_factor'] = 1.6
                optim_kwargs['balance_scale_factor'] = 2.4
            elif args.model == 'sdxl':
                optim_kwargs['guidance_scale_factor'] = 1.8
                optim_kwargs['balance_scale_factor'] = 2.6
        elif args.method == 'sag':
            optim_kwargs['sag_scale'] = 0.75
        elif args.method == 'seg':
            optim_kwargs['seg_scale'] = 3.0
            optim_kwargs['seg_blur_sigma'] = 100.0
            optim_kwargs['seg_applied_layers'] = ['mid']


        

        output = pipe.forward_optim(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=args.inference_step,
            latents=start_latents,
            return_dict=False,
            **optim_kwargs
        )
        optim_img = output[0][0]

        print(len(pipe.guidance_scale_list))
        print(pipe.guidance_scale_list)
        guidance_scale_list = pipe.guidance_scale_list
        guidance_scale_list = [x for x in guidance_scale_list if x == x] # remove NaN
        # mean_scale = torch.mean(torch.Tensor(pipe.guidance_scale_list))
        # interpolated_list = F.interpolate(torch.Tensor(pipe.guidance_scale_list).unsqueeze(0).unsqueeze(0) / 3, 
        #                                          size=args.inference_step*3,
        #                                          mode='linear', 
        #                                          align_corners=False)
        mean_value = torch.Tensor(guidance_scale_list).mean().item() if len(pipe.guidance_scale_list) != 0 else args.denoising_cfg
        if args.method == 'freeu' and args.model == 'sd21':
            if mean_value < args.denoising_cfg:
                mean_value *= 1.4
        # if nfe_ratio > 1:
        #     pipe.guidance_scale_list = [min(max(guidance_scale, mean_value), guidance_scale * nfe_ratio)] * int(args.inference_step * nfe_ratio)
        # else:
        pipe.guidance_scale_list = [mean_value] * int(args.inference_step * nfe_ratio)
        eq_guidance_scale = pipe.guidance_scale_list[0]

        original_kwargs = dict()
        if args.method == 'zigzag':
            pass
        elif args.method == 'cfgpp':
            guidance_scale = args.denoising_cfg
        elif args.method == 'pag':
            original_kwargs['pag_scale'] = 0
        elif args.method == 'w2sd':
            pipe.enable_lora()
            pipe.set_adapters(lora_name, adapter_weights=args.strong_lora_scale)
        elif args.method == 'apg':
            original_kwargs['adaptive_projected_guidance'] = False
        elif args.method == 'freeu':
            # register_free_upblock2d(pipe, b1=1, b2=1, s1=1, s2=1) # type: ignore
            # register_free_crossattn_upblock2d(pipe, b1=1, b2=1, s1=1, s2=1) # type: ignore
            original_kwargs['do_freeu'] = False
        elif args.method == 'sag':
            original_kwargs['sag_scale'] = 0
        elif args.method == 'seg':
            original_kwargs['seg_scale'] = 0
            original_kwargs['seg_blur_sigma'] = 0
            # original_kwargs['seg_applied_layers'] = []
        
        output = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            latents=start_latents,
            return_dict=False,
            num_inference_steps=int(args.inference_step * nfe_ratio),
            **original_kwargs
        )
        original_img = output[0][0]

        pipe.guidance_scale_list = []

        optim_hpsv2_score = evaluator.hpsv2_score(prompt, optim_img)
        original_hpsv2_score = evaluator.hpsv2_score(prompt, original_img)
        print("optim_hpsv2_score: ", optim_hpsv2_score, "original_hpsv2_score: ", original_hpsv2_score)
        if optim_hpsv2_score > original_hpsv2_score:
            print("Win this prompt: ", prompt)
            WINNING_NUMBER += 1
        OPTIM_SCORE += optim_hpsv2_score
        ORIGINAL_SCORE += original_hpsv2_score
        
        optim_img.save(os.path.join(base_dir, 'optim', 'new{:07d}.png'.format(idx)))
        original_img.save(os.path.join(base_dir, 'optim', 'original{:07d}.png'.format(idx)))
        data = {
            'index': idx,
            'caption': prompt,
            'optimized_score_list': optim_hpsv2_score,
            'original_score_list': original_hpsv2_score,
            'mean_value': mean_value,
            'eq_guidance_scale': eq_guidance_scale,
            'guidance_scale_list': guidance_scale_list
        }
        data['aes_optim'] = evaluator.aes_score(optim_img)
        data['aes_original'] = evaluator.aes_score(original_img)

        data['pick_optim'] = evaluator.pick_score(prompt, optim_img)
        data['pick_original'] = evaluator.pick_score(prompt, original_img)

        ir_optim, ir_original = evaluator.image_reward(prompt, optim_img, original_img)
        data['ir_optim'] = ir_optim
        data['ir_original'] = ir_original

        clip_optim = evaluator.clip_score(optim_img, prompt)
        clip_original = evaluator.clip_score(original_img, prompt)
        data['clip_optim'] = clip_optim
        data['clip_original'] = clip_original

        preference = evaluator.mps_score(original_img, optim_img, prompt)
        mps_original, mps_optim = preference[0], preference[1]
        data['mps_optim'] = mps_optim
        data['mps_original'] = mps_original

        with open(os.path.join(base_dir, 'json', 'new{:07d}.json'.format(idx)), 'w+') as file:
            json.dump(data, file)
            file.write('\n')
        
        _ += 1

    print("Winning Rate : ", WINNING_NUMBER / (args.end_idx - args.start_idx))
    print("Optimized Score : ", OPTIM_SCORE / (args.end_idx - args.start_idx))
    print("Original Score : ", ORIGINAL_SCORE / (args.end_idx - args.start_idx))
