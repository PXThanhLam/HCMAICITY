import sys
sys.path.append('Drone_FasterRCNN')
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
import time
import os
config_file = "drone_demo/e2e_faster_rcnn_X_101_32x8d_FPN_1x_visdrone.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.merge_from_list(["MODEL.WEIGHT", "drone_demo/visdrone_model_0360000.pth"])

coco_demo = COCODemo(
    cfg,
    min_image_size=300,
    confidence_threshold=0.2,
)
# load image and then run prediction
for img_path in os.listdir('test_img'):
    start=time.time()
    image = cv2.imread('test_img/'+img_path)
    predictions = coco_demo.run_on_opencv_image(image)
    print(time.time()-start)
    cv2.imwrite('test_result/'+img_path, predictions)
    
