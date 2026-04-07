
import os
from T2IBenchmark import calculate_fid
from tqdm import tqdm
from glob import glob
import json

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

base_dir = '/data/root/cfg_DiT_imagenet/results/ilsvrc2012_cfgpp/output_ditxl2_1000_eqcfg_True'
imgnet_dir = '/data/shared_data/ILSVRC2012/val'

# all_original = []
# all_new = []

# pbar = tqdm(total=len(labels))
# for img_class in labels:
#     class_label = img_class['label']

#     gt_img = os.path.join(imgnet_dir, class_label)
#     original_img = os.path.join(base_dir, class_label, 'original')
#     new_img = os.path.join(base_dir, class_label, 'new')

#     # print('ss=------------')
#     # print(new_img, gt_img)

#     fid_new, _ = calculate_fid(
#         new_img,
#         gt_img
#     )

#     fid_original, _ = calculate_fid(
#         original_img,
#         gt_img
#     )

#     all_original.append(fid_original)
#     all_new.append(fid_new)

#     pbar.update(1)

# print(base_dir)
# print('original_fid:', sum(all_original) / len(labels))
# print('new_fid:', sum(all_new) / len(labels))

gt_img = []
original_img = []
new_img = []
json_list = []

for img_class in labels:
    class_label = img_class['label']
    gt_img += list(glob(os.path.join(imgnet_dir, class_label, '*.*')))
    original_img += list(glob(os.path.join(base_dir, class_label, 'original', '*.*')))
    new_img += list(glob(os.path.join(base_dir, class_label, 'new', '*.*')))
    for i in range(len(glob(os.path.join(base_dir, class_label, 'json/*.json')))):
        with open(os.path.join(base_dir, class_label, 'json', 'new{:07d}.json'.format(i)), 'r') as f:
            json_list.append(json.load(f))
eq_cfg = sum([x['eq_guidance_scale'] for x in json_list]) / len(json_list)
print('eq_cfg:', eq_cfg)

fid_new, _ = calculate_fid(
    new_img,
    gt_img
)
fid_original, _ = calculate_fid(
    original_img,
    gt_img
)
print(base_dir)
print('original_fid:', fid_original)
print('new_fid:', fid_new)

