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
from convert_cocoformat import convert_coco

compound_coef =3
force_input_size = None  # set None to use default size


img_roots='phone_data/sp'
img_paths=[]
for img_root in os.listdir(img_roots):
    img_root=img_roots+'/'+img_root
    for img_path in os.listdir(img_root):
        if img_path.endswith('jpg'):
            img_paths.append(img_root+'/'+img_path)
print('DATA LOADED')
print(len(img_paths))
# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2
batch_eval=1

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


color_list = standard_to_bgr(STANDARD_COLORS)
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()
print('MODEL LOADED')
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()
def display(names,preds, imgs, imshow=True, imwrite=False,save_bbox=True):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue
        f = open('test/'+names[i]+'.txt', "w")
        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])
            if save_bbox and obj=='cell phone':
                f.write(str(67)+' '+str(score)+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+'\n')

        if imshow:
            cv2.imshow(names[i], imgs[i])
            cv2.waitKey(0)
        if imwrite:
            cv2.imwrite('test/'+names[i]+'.jpg', imgs[i])
            

with torch.no_grad():
    for indx in range((len(img_paths)+batch_eval-1)//batch_eval):
        print(indx*batch_eval)
        ori_img_batch,framed_img_batch,metas_batch=preprocess(
            img_paths[indx*batch_eval:min((indx+1)*batch_eval,len(img_paths))], max_size=input_size)
        img_names = img_paths[indx*batch_eval:min((indx+1)*batch_eval,len(img_paths))]
        
        convert_coco(img_names)

        image_names=[]
        for img_name in img_names:
            image_names.append(img_name.replace('/','_')[:-4])

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_img_batch], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_img_batch], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)
        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
        out = invert_affine(metas_batch, out)
        display(image_names,out, ori_img_batch, imshow=False, imwrite=True)