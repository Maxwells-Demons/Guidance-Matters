
from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats
import os
from glob import glob
import json

base_dir = '/data/root/cfg_bench_coco/results/coco_cfgpp/output_sdxl_1000_eqcfg_True'

json_list = []

for i in range(len(glob(os.path.join(base_dir, 'json/*.json')))):
    with open(os.path.join(base_dir, 'json', 'new{:07d}.json'.format(i)), 'r') as f:
        json_list.append(json.load(f))
eq_cfg = sum([x['eq_guidance_scale'] for x in json_list]) / len(json_list)
print('eq_cfg:', eq_cfg)
print('aes_original:', sum([x['aes_original'] for x in json_list]) / len(json_list))
print('aes_optim:', sum([x['aes_optim'] for x in json_list]) / len(json_list))
print('hpsv2_original:', sum([x['original_score_list'] for x in json_list]) / len(json_list))
print('hpsv2_optim:', sum([x['optimized_score_list'] for x in json_list]) / len(json_list))

fid_new, _ = calculate_fid(
    os.path.join(base_dir, 'new'),
    get_coco_fid_stats()
)
fid_original, _ = calculate_fid(
    os.path.join(base_dir, 'original'),
    get_coco_fid_stats()
)

print(base_dir)
print('fid_new', fid_new)
print('fid_original', fid_original)