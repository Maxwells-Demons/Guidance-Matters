
import os
import torch
import argparse
import json
from accelerate.utils import set_seed
from diffusers.utils.torch_utils import randn_tensor
from glob import glob
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2
from PIL import Image

from utils import Evaluator

def get_args():
    # pick: test_unique_caption_zh.csv       draw: drawbench.csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='ditxl2', type=str)
    parser.add_argument("--method", default='zigzag', type=str)
    parser.add_argument("--inference_step", default=50, type=int)
    parser.add_argument("--size", default=256, type=int)
    parser.add_argument("--T_max", default=1, type=int)
    parser.add_argument("--seed", default=1000, type=int)
    parser.add_argument("--RatioT", default=1.0, type=float)    #if RatioT==1,则退化为start point优化
    parser.add_argument("--denoising_cfg", default=4, type=float)
    parser.add_argument("--inversion_cfg", default=0, type=float)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=None, type=int)
    parser.add_argument("--class_id", default=None, type=int)
    parser.add_argument("--img_num", default=None, type=int)
    parser.add_argument("--dataset", default='ilsvrc2012', type=str)
    # parser.add_argument("--prompt_path", default='datasets/test_unique_caption_zh.csv', type=str)
    # parser.add_argument("--dataset", default='coco', type=str)

    parser.add_argument("--do_eq_cfg", default=True, type=bool)
    
    args =  parser.parse_args()
    return args




labels = []
with open('./imagenet_label') as f:
    lines = f.readlines()
    idx = 0
    for line in lines:
        if len(line) < 5:
            continue
        line_split = line.strip().split(',')
        label_name = ' '.join(line_split[1:])
        label = line_split[0].split(' ')[1]
        # labels[idx] = label_name
        labels.append({
            'label_id': idx,
            'name': label_name,
            'label': label
        })
        idx += 1



if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)
    device = torch.device('cuda')

    # dtype = torch.float16
    args = get_args()
    print("args.seed: ", args.seed)
    set_seed(args.seed)

    base_dir = f'./results/{args.dataset}_{args.method}/output_{args.model}_{args.seed}_eqcfg_{args.do_eq_cfg}/'
    class_label = labels[int(args.class_id)]['label']

    if args.end_idx is None:
        args.end_idx = args.img_num
    if args.start_idx is None:
        args.start_idx = 0
    
    finished = True
    if args.start_idx is not None and args.end_idx is not None:
        for idx in range(args.start_idx, args.end_idx):
            if os.path.exists(os.path.join(base_dir, class_label, 'json', 'new{:07d}.json'.format(idx))):
                continue
            else:
                finished = False
                break
    if finished:
        print(f'({args.start_idx}, {args.end_idx}) finished.')
        exit()
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, class_label, 'new'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, class_label, 'original'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, class_label, 'json'), exist_ok=True)

    image_size = args.size #@param [256, 512]
    vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    latent_size = int(image_size) // 8
    # Load model:
    model = DiT_XL_2(input_size=latent_size).to(device)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval() # important!
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)

    diffusion = create_diffusion(str(args.inference_step))
    evaluator = Evaluator()

    _ = 0
    idx_list = list(range(args.start_idx, args.end_idx))
    # random.shuffle(idx_list)
    prompt = f"a photo of {labels[int(args.class_id)]['name']}"
    for idx in idx_list:
        diffusion.guidance_scale_list = []

        print('idx:', idx)
        print('prompt:', prompt)
        # print(os.path.join(base_dir, 'json', 'new{:07d}.json'.format(idx)))
        if os.path.exists(os.path.join(base_dir, class_label, 'json', 'new{:07d}.json'.format(idx))):
            print("pass this prompt (existed): ", prompt)
            continue
        
        n = 1
        z = randn_tensor((n, 4, latent_size, latent_size), device=device, generator=torch.Generator('cuda').manual_seed(args.seed+idx))
        y = torch.tensor([args.class_id], device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.denoising_cfg)

        # Sample images:
        diffusion.method = args.method
        if diffusion.method == 'cfgpp':
            model_kwargs['cfg_scale'] = 0.4
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, 
            model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        optim_img = vae.decode(samples / 0.18215).sample
        save_image(optim_img, os.path.join(base_dir, class_label, 'new', 'new{:07d}.png'.format(idx)), nrow=int(1), 
           normalize=True, value_range=(-1, 1))

        guidance_scale_list = diffusion.guidance_scale_list
        eq_cfg_scale = sum(guidance_scale_list) / len(guidance_scale_list)

        
        diffusion.method = 'cfg'
        if diffusion.method == 'cfg':
            model_kwargs['cfg_scale'] = 4
        if args.do_eq_cfg:
            print('eq_cfg_scale:', eq_cfg_scale)
            model_kwargs = dict(y=y, cfg_scale=eq_cfg_scale)
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, 
            model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        original_img = vae.decode(samples / 0.18215).sample
        save_image(original_img, os.path.join(base_dir, class_label, 'original', 'original{:07d}.png'.format(idx)), nrow=int(1), 
           normalize=True, value_range=(-1, 1))

        optim_img = Image.open(os.path.join(base_dir, class_label, 'new', 'new{:07d}.png'.format(idx)))
        original_img = Image.open(os.path.join(base_dir, class_label, 'original', 'original{:07d}.png'.format(idx)))
        optim_hpsv2_score = evaluator.hpsv2_score(prompt, optim_img)
        original_hpsv2_score = evaluator.hpsv2_score(prompt, original_img)
        print("optim_hpsv2_score: ", optim_hpsv2_score, "original_hpsv2_score: ", original_hpsv2_score)
        
        
        data = {
            'index': idx,
            'caption': prompt,
            'optimized_score_list': optim_hpsv2_score,
            'original_score_list': original_hpsv2_score,
            'eq_guidance_scale': eq_cfg_scale,
            'guidance_scale_list': guidance_scale_list,
            'label': labels[int(args.class_id)]
        }
        # print(evaluator.aes_model.clip.device)
        # print(optim_img.device)
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

        with open(os.path.join(base_dir, class_label, 'json', 'new{:07d}.json'.format(idx)), 'w+') as file:
            json.dump(data, file)
            file.write('\n')
        
        _ += 1
    

