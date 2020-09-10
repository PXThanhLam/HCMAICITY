# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import os
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

img_root='TestVehicle'
compound_coef = 7
force_input_size = None  # set None to use default size
device = torch.device('cuda:0')

img_paths=[]
for img_path in os.listdir(img_root):
    img_paths.append(img_root+'/'+img_path)
list_img_paths=[[img_path] for img_path in img_paths]
# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.5

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list =['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

obj_interest=[ 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']

color_list = standard_to_bgr(STANDARD_COLORS)

def display(preds, imgs, imshow=True, imwrite=False,start=0):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            if obj in obj_interest:
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(img_root+'Result'+f'/img_inferred_d{compound_coef}_this_repo_{i+start}.jpg', imgs[i])

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
model.requires_grad_(False)
model.eval()
if use_cuda:
    model = model.to(device)
if use_float16:
    model = model.half()
print('MODEL LOADED')
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()


start_index=0
for img_paths in list_img_paths:
    start=time.time()
    ori_imgs = [cv2.imread(img_path) for img_path in img_paths]
    ori_imgs, framed_imgs, framed_metas = preprocess(ori_imgs, max_size=input_size)
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).to(device) for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)


    with torch.no_grad():
        inference_start=time.time()
        features, regression, classification, anchors = model(x)
        print('Inference time :'+str((time.time()-inference_start)/len(img_paths)))
        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
   
    out = invert_affine(framed_metas, out)
    print('processing time : '+str((time.time()-start)/len(img_paths)))
    display(out, ori_imgs, imshow=False, imwrite=True,start=start_index)
    start_index+=len(img_paths)   

