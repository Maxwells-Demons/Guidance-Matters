
import torch
from transformers import CLIPProcessor, CLIPModel
import csv
import json
import torch.nn as nn
import ImageReward as reward
from transformers import AutoProcessor, AutoModel
from pycocotools.coco import COCO

from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy.linalg import sqrtm

from transformers import CLIPImageProcessor, AutoTokenizer
from PIL import Image
from io import BytesIO

device = torch.device('cuda')


class Evaluator:
    def __init__(self):
        self.hpsv2 = HPSV2()

        self.ir_model = reward.load("ImageReward-v1.0")
        self.aes_model = AestheticScorer(dtype=torch.float32)

        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

        self.pick_processor = AutoProcessor.from_pretrained(processor_name_or_path, local_files_only=True)
        self.pick_model = AutoModel.from_pretrained(model_pretrained_name_or_path, local_files_only=True).eval().to(device)
        self.clip_model = CLIPScore()
        self.mp_model = MPScore()

        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16', local_files_only=True).eval().to(device)
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16', local_files_only=True)
    
    @torch.no_grad()
    def hpsv2_score(self, prompt, img):
        return float(self.hpsv2.score(prompt, img)[0])
    
    @torch.no_grad()
    def pick_score(self, prompt, images):
        # preprocess
        image_inputs = self.pick_processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        text_inputs = self.pick_processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)


        with torch.no_grad():
            # embed
            image_embs = self.pick_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = self.pick_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = self.pick_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # get probabilities if you have multiple images to choose from
            probs = torch.softmax(scores, dim=-1)
        
        return scores.cpu().tolist()[0]
    
    @torch.no_grad()
    def aes_score(self, image):
        return self.aes_model(image).item()
    
    @torch.no_grad()
    def image_reward(self, prompt, optim_img, original_img):
        ranking, rewards = self.ir_model.inference_rank(prompt, [optim_img, original_img])
        ir_optim, ir_original = rewards
        return ir_optim, ir_original

    @torch.no_grad()
    def mps_score(self, original_image, optimized_image, prompt):
        return self.mp_model.score(original_image, optimized_image, prompt)

    @torch.no_grad()
    def clip_score(self, images, prompts):
        image_inputs = self.clip_processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        text_inputs = self.clip_processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)


        with torch.no_grad():
            # embed
            image_embs = self.clip_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = self.clip_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = self.clip_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # get probabilities if you have multiple images to choose from
            probs = torch.softmax(scores, dim=-1)
        
        return scores.cpu().tolist()[0]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)

class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14', local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', local_files_only=True)
        self.mlp = MLP()
        state_dict = torch.load("/data/shared_data/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth")
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        # images = transforms.ToTensor()(images)
        # images = (images * 255).round().clamp(0, 255).to(torch.uint8)

        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

class CLIPScore:
    def __init__(self):
        from torchmetrics.functional.multimodal import clip_score
        from functools import partial
        self.clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def calculate_clip_score(self, images, prompts):
        images_int = (images * 255).astype("uint8")
        clip_score = self.clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)

    def score(self, images, prompts):
        if images.ndim == 3:
            images = images[None, ...]
        # print(images.shape)
        return self.calculate_clip_score(images, [prompts])


class MPScore:
    def __init__(self):
        condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things." 
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        image_processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)
        model_ckpt_path = "ckpt/MPS_overall_checkpoint.pth"
        model = torch.load(model_ckpt_path, map_location='cpu')
        model.model.text_model.eos_token_id=2
        model.eval().to(device)
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.condition = condition

    def _process_image(self, image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        if isinstance(image, str):
            image = Image.open( image )
        image = image.convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values
    
    def _tokenize(self, caption):
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids
    
    @torch.no_grad()
    def _compute_score(self, images, caption):
        image_inputs = torch.concatenate([self._process_image(images[0]).to(device), self._process_image(images[1]).to(device)])
        text_inputs = self._tokenize(caption).to(device)
        condition_inputs = self._tokenize(self.condition).to(device)
        text_features, image_0_features, image_1_features = self.model(text_inputs, image_inputs, condition_inputs)
        image_0_features = image_0_features / image_0_features.norm(dim=-1, keepdim=True)
        image_1_features = image_1_features / image_1_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_0_scores = self.model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_0_features))
        image_1_scores = self.model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_1_features))
        scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
        probs = torch.softmax(scores, dim=-1)[0]
        return probs.cpu().tolist()
    
    def score(self, original_image, optimized_image, prompt):
        return self._compute_score([original_image, optimized_image], prompt)
    

def get_inception_model():
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    return model

class FIDScoreEvaluator:
    def __init__(self):
        self.model = get_inception_model()
        self.transform = transforms.Compose([
            transforms.Resize(299),  # InceptionV3 要求 299x299 大小的图像
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def calculate_fid(self, real_img_paths, generated_img_paths):
        real_features = self.get_features(real_img_paths)
        generated_features = self.get_features(generated_img_paths)

        mu_real = np.mean(real_features, axis=0)
        mu_generated = np.mean(generated_features, axis=0)

        sigma_real = np.cov(real_features, rowvar=False)
        sigma_generated = np.cov(generated_features, rowvar=False)

        diff = mu_real - mu_generated
        covmean = sqrtm(sigma_real.dot(sigma_generated))

        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = np.sum(diff**2) + np.trace(sigma_real + sigma_generated - 2 * covmean)
        return fid

    
    def get_features(self, img_paths):
        features = []
        with torch.no_grad():
            for img_path in tqdm(img_paths):
                image = self.transform(Image.open(img_path)).unsqueeze(0)
                feature = self.model(image)
                features.append(feature.squeeze(0).cpu().numpy())
        return np.array(features)


class HPSV2:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("adams-story/HPSv2-hf", local_files_only=True).to(device)
        self.processor = CLIPProcessor.from_pretrained("adams-story/HPSv2-hf", local_files_only=True)
        # model.logit_scale = 4.6055
        # e**4.6055 ~= 100

    @torch.no_grad()
    def score(self, prompt, image):
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt", max_length=77, padding='max_length', truncation=True).to(device)
        hps_out = self.model(**inputs)

        return hps_out.logits_per_image

def load_coco_prompt(path='datasets/coco/captions_val2017.json', length=200):
    coco = COCO(path)
    coco_img_ids = coco.getImgIds()[:length]
    prompts = []
    for img_id in coco_img_ids:
        prompts.append(coco.imgToAnns[img_id][0]['caption'])
    return prompts

def load_prompt(path, seed_path="datasets/HPD/HPD_prompt2seed.json", prompt_version="hpsv2"):
    if prompt_version == 'pick':
        prompts = []
        with open(path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == "caption":
                    continue
                prompts.append(row[1])

        prompts = prompts[0:101]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)

        return prompts, seed_list
    
    elif prompt_version == 'draw':
        prompts = []
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == "Prompts":
                    continue
                prompts.append(row[0])

        prompts = prompts[0:200]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)

        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)
        return prompts, seed_list
    elif prompt_version == 'challengebench':
        prompts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                new_prompt = line.strip()
                prompts.append(new_prompt)
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list
        print("prompts: ", len(prompts))
        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)    
        return prompts, seed_list
    else:
        prompts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                parts = line.strip().split(', ')
                new_prompt = ""
                for i in range(1, len(parts)-1):
                    if i == len(parts)-2:
                        new_prompt += parts[i]
                    else:
                        new_prompt += parts[i] + ", "
                prompts.append(new_prompt)
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list
        print("prompts: ", len(prompts))
        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)

        return prompts, seed_list