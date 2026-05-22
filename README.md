# Guidance Matters

Official code for **"Guidance Matters: Rethinking the Evaluation Pitfall for Text-to-Image Generation."**

Many recent guidance methods for diffusion-based T2I models — Zigzag, PAG, SAG, SEG, CFG++, FreeU, APG, TDG, W2SD — report higher HPSv2 / PickScore / ImageReward / FID than vanilla CFG. This repo's claim: a large fraction of those gains come from **implicitly inflating the effective CFG strength** (and/or burning extra NFEs), not from a fundamentally better guidance signal. Once you re-express each method's per-step perturbation in CFG units and let vanilla CFG run at the same effective strength under the same NFE budget, much of the reported advantage disappears.

The codebase implements that fair-comparison protocol end-to-end across SD 2.1, SDXL, SD 3.5, and class-conditional DiT-XL/2.

---

## The protocol in one paragraph

For each prompt:

1. **Run the candidate method.** At every denoising step, log
   `eq_cfg_t = ‖v_p‖ / ‖ε_text − ε_uncond‖`,
   where `v_p` is the actual perturbation that method adds to the predicted noise. This re-expresses any guidance-shaped change as an equivalent CFG coefficient.
2. **Average** `eq_cfg_t` over the trajectory → `mean_eq_cfg`.
3. **Run vanilla CFG** at `guidance_scale = mean_eq_cfg` for `inference_step × nfe_ratio` steps, where `nfe_ratio` matches the method's extra forward passes (zigzag = 3, PAG/SAG/TDG/SEG = 1.5, CFG++/APG/FreeU = 1).
4. **Score both images** with HPSv2, PickScore, ImageReward, Aesthetic, CLIP, MPS (and FID for COCO / ImageNet). Report winning rate and per-metric gap.

The key formula is implemented inside each modified pipeline, e.g. `pipelines/sdxl_pag.py:1266`, `pipelines/sdxl_cfgpp.py:1826`, `pipelines/sdxl_zigzag_1.py:2469`. The two-pass loop lives in `inference.py`.

---

## Repository layout

```
.
├── inference.py              # Main eval loop on Pick-a-Pic / DrawBench / HPDv2 / COCO
├── utils.py                  # Evaluator: HPSv2, ImageReward, Aesthetic, PickScore, CLIP, MPS, FID
├── pipelines/                # Per-method, per-backbone re-implementations that log eq_cfg per step
│   ├── sdxl_zigzag_1.py  sdxl_pag.py  sdxl_cfgpp.py  sdxl_apg.py
│   ├── sdxl_freeu.py     sdxl_sag.py  sdxl_seg.py    sdxl_tdg.py   sdxl_w2sd.py
│   ├── sd_*.py           # SD 2.1 variants
│   └── sd35_*.py         # SD 3.5 variants
├── dataset/
│   ├── test_unique_caption_zh.csv     # Pick-a-Pic test captions
│   ├── drawbench.csv                  # DrawBench
│   └── HPD/                           # HPDv2 prompts + prompt→seed map
├── coco/                     # COCO-30K eval + FID against COCO val stats (T2IBenchmark)
│   ├── inference.py  calc_fid.py  pipelines/  T2IBenchmark/
├── imagenet/                 # Same protocol for class-conditional DiT-XL/2 on ImageNet
│   ├── sample_ddp.py  sample.py  train.py  models.py  diffusion/
│   ├── calc_fid.py   T2IBenchmark/
│   └── imagenet_label  download.py
└── trainer/models/           # CLIP / cross-modal encoders used by metric models
```

Each `pipelines/<backbone>_<method>.py` exposes the same surface used by `inference.py`:

- a `forward_optim` (or `__call__`) that runs the method and appends per-step `eq_cfg` to `pipe.guidance_scale_list`;
- a vanilla `__call__` path used for the matched-CFG, matched-NFE rerun (toggled by setting method-specific kwargs to their no-op values, e.g. `pag_scale=0`, `sag_scale=0`, `do_freeu=False`).

---

## Supported (model × method) matrix

|              | zigzag | cfgpp | pag | sag | seg | apg | freeu | tdg | w2sd |
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **SD 2.1**   | ✓ | ✓ | ✓ | ✓ |   | ✓ | ✓ |   |   |
| **SDXL**     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **SD 3.5**   | ✓ | ✓ | ✓ |   |   | ✓ |   | ✓ |   |
| **DiT-XL/2** | see `imagenet/` |

`nfe_ratio` per method (set inside `pipe_handler`):

| method | nfe_ratio | rationale |
|---|---|---|
| `cfgpp`, `apg`, `freeu` | 1.0 | no extra forwards beyond CFG |
| `pag`, `sag`, `seg`, `tdg` | 1.5 | one extra perturbed forward each step |
| `zigzag`, `w2sd` | 3.0 | inversion + re-denoise loop |

---

## Setup

Tested on Linux + CUDA, PyTorch ≥ 2.1, `diffusers` ≥ 0.30.

```bash
pip install torch torchvision diffusers transformers accelerate \
            ImageReward image-reward pycocotools scipy tqdm bitsandbytes \
            torchmetrics pillow
```

Model checkpoints expected on disk:

| purpose | path / hub id |
|---|---|
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` |
| SD 2.1 | `stabilityai/stable-diffusion-2-1` |
| SD 3.5 | `/data/shared_data/SDV35/sdv35/` (4-bit NF4 transformer loaded via `BitsAndBytesConfig`) |
| HPSv2 | `adams-story/HPSv2-hf` |
| ImageReward | `ImageReward-v1.0` |
| Aesthetic predictor | `/data/shared_data/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth` |
| PickScore | `yuvalkirstain/PickScore_v1` (+ `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` processor) |
| MPS | `ckpt/MPS_overall_checkpoint.pth` |
| W2SD LoRA (SDXL only) | `./ckpt/xlMoreArtFullV1.pREw.safetensors` |

`utils.py` loads the metric models with `local_files_only=True`, so populate the HF cache (or change the flag) before running.

---

## Running the main benchmark

```bash
# SDXL + Zigzag, evaluated on Pick-a-Pic
python inference.py \
    --model sdxl --method zigzag \
    --dataset pick \
    --inference_step 50 --denoising_cfg 5.5 \
    --size 1024 --seed 1000

# SD 3.5 + PAG, evaluated on DrawBench
python inference.py --model sd35 --method pag --dataset draw

# SD 2.1 + CFG++, evaluated on HPDv2
python inference.py --model sd21 --method cfgpp --dataset hpd
```

Datasets recognized by `--dataset`: `pick` (Pick-a-Pic), `draw` (DrawBench), `hpd` (HPDv2), `coco` (200 captions from COCO val). Custom prompt lists go via `--prompt_path`.

Outputs land in `./results/<dataset>_<method>/output_<model>_train_<seed>/`:

- `optim/new{idx}.png` — image from the candidate method
- `optim/original{idx}.png` — image from vanilla CFG at the matched effective strength
- `json/new{idx}.json` — per-prompt scores plus `mean_value` (the recovered equivalent CFG) and the full `guidance_scale_list`

The script prints aggregate `Winning Rate`, `Optimized Score`, `Original Score` (HPSv2) at the end. The JSON files retain the other six metrics for offline aggregation.

### Method-specific knobs

| flag | applies to | default | notes |
|---|---|---|---|
| `--inversion_cfg` | zigzag | 0 | CFG used during inversion sub-loop |
| `--lora_path`, `--lora_sclae`, `--strong_lora_scale`, `--weak_lora_scale`, `--strong_guidance_scale`, `--weak_guidance_scale` | w2sd | see `argparse` | gap-LoRA + gap-CFG pair for W2SD |

Other knobs (`pag_scale=3`, `sag_scale=0.75`, `seg_scale=3.0`, FreeU's `b1/b2/s1/s2`, TDG's `guidance_scale_factor` / `balance_scale_factor`) are hard-coded inside `inference.py` to the values used in the paper, indexed by `(model, method)`.

---

## COCO FID

```bash
# 1) generate paired images (optim vs. matched-CFG vanilla) on COCO captions
python coco/inference.py --model sdxl --method cfgpp --dataset coco

# 2) compute FID against COCO val stats for both folders
python coco/calc_fid.py
```

`coco/calc_fid.py` reads `eq_guidance_scale`, `aes_*`, `hpsv2_*` from the per-prompt JSONs and prints the average alongside the two FIDs.

## ImageNet (DiT-XL/2)

`imagenet/` ports the protocol to class-conditional DiT-XL/2 to show the pitfall is not text-conditional in nature.

```bash
torchrun --nproc_per_node=8 imagenet/sample_ddp.py [--num-fid-samples 50000 ...]
python imagenet/calc_fid.py
```

`sample_ddp.py` produces a `.npz` compatible with the ADM FID evaluation tooling.

---

## Output JSON schema

Each `json/new{idx}.json` written by `inference.py`:

```jsonc
{
  "index": 17,
  "caption": "...",
  "optimized_score_list": 0.273,        // HPSv2 of candidate
  "original_score_list": 0.268,         // HPSv2 of matched-CFG vanilla
  "mean_value": 6.42,                   // recovered equivalent CFG
  "eq_guidance_scale": 6.42,
  "guidance_scale_list": [ ... ],       // per-step eq_cfg trajectory
  "aes_optim": ...,    "aes_original": ...,
  "pick_optim": ...,   "pick_original": ...,
  "ir_optim": ...,     "ir_original": ...,
  "clip_optim": ...,   "clip_original": ...,
  "mps_optim": ...,    "mps_original": ...
}
```

`mean_value` is the headline number — it is what each guidance method is "really" paying in CFG-units.

---

## Adding a new method

1. Copy the closest existing pipeline (e.g. `pipelines/sdxl_pag.py`) and implement your method's forward.
2. At every step, after computing your guidance vector `v_p`, append
   `(‖v_p‖ / ‖ε_text − ε_uncond‖).item()` to `self.guidance_scale_list`.
3. Make sure the same pipeline class also exposes a vanilla `__call__` path (or accepts a kwarg that turns the method into a no-op) so the matched-CFG rerun can use it.
4. Add a branch in `pipe_handler` in `inference.py` and pick the right `nfe_ratio` for your method's extra forwards.

If your method is not strictly a "perturbation on top of CFG" (e.g. it changes the score vector in a non-additive way), pick the projection of your update onto `(ε_text − ε_uncond)` direction before taking the norm — that is what makes the comparison apples-to-apples.

---

## Citation

```bibtex
@article{guidance_matters,
  title  = {Guidance Matters: Rethinking the Evaluation Pitfall for Text-to-Image Generation},
  author = {Dian Xie, et al.},
  year   = {2025}
}
```
