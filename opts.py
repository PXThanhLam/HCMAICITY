from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
   

    # system
    self.parser.add_argument('--gpus', default='0, 1',
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=8,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    self.parser.add_argument('--vis_thresh', type=float, default=0.5,
                             help='visualization threshold.')
    
    # # model
    # self.parser.add_argument('--detection_config', default='Pedestron/configs/elephant/cityperson/cascade_hrnet.py', help='person detection config file')
    # self.parser.add_argument('--detection_path', default='Pedestron/models_pretrained/CrowdHuman2.pth.stu', help='person detection path weights')
    # self.parser.add_argument('--reid_model', default='OSNet', help='reid model usage')
    self.parser.add_argument('--detection_model', default='Efficient', help='FasterRcnn or EfficientDet') #Efficient
    self.parser.add_argument('--min_img_size', default=370, help='min image size of FasterRcnn')




    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.')
    

    # test
    self.parser.add_argument('--K', type=int, default=128,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--fix_res', action='store_true',
                             help='fix testing resolution or keep '
                                  'the original resolution')
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')
    # tracking
    #for effdet:
    #cam9 0.12 #cam4 0.27 #cam5 0.1 #cam6 0.15, #cam7 0.25 #cam8 0.08 #cam11 0.25 #cam12 0.15 #cam 13 0.15 #cam14 0.2 
    #cam 15 0.2 #cam16 0.11 #cam17 0.12 #cam1 0.15 #cam2 0.15 #cam3 0.2 #cam10 0.15 #cam 18,23,24 0.2

    #for faster
    self.parser.add_argument('--conf_thres', type=float, default=0.15, help='confidence thresh for tracking')
    self.parser.add_argument('--det_thres', type=float, default=0.15, help='confidence thresh for detection') 

    #for effdet
    #cam1 0.3

    #for faster
    self.parser.add_argument('--near_cam_thres', type=float, default=0.3, help='confidence thresh for detection')
    #for effdet
    #cam1 0.35

    #for faster
    self.parser.add_argument('--near_cam_big_veh_thres', type=float, default=0.35, help='confidence thresh for detection')
    #for effdet
    #cam1 0.2

    #for faster
    self.parser.add_argument('--big_veh_thres', type=float, default=0.2, help='confidence thresh for detection')

    self.parser.add_argument('--cam_id', type=str, default='09', help='inference batch size')
    self.parser.add_argument('--inference_batch_size', type=float, default=32, help='inference batch size')
    self.parser.add_argument('--nms_thres', type=float, default=0.5, help='iou thresh for nms') #cam4,5,6,7,8,9,11,13,14  0.5
    self.parser.add_argument('--track_buffer', type=int, default=15, help='tracking buffer') #cam4,5,6,7,8,9,11,13,14   15 #cam20 18 #cam23 9
    self.parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    self.parser.add_argument('--input-video', type=str, default='', help='path to the input video')
    self.parser.add_argument('--input-meta', type=str, default='', help='path to the polygon and moi of video')

    self.parser.add_argument('--output-format', type=str, default='video', help='video or text')
    self.parser.add_argument('--output-root', type=str, default='', help='expected output root path')
    self.parser.add_argument('--compound_coef', type=int, default=3, help='compound_coef of efficientdet')

    self.parser.add_argument('--cam_list', type=str, default="01", help='')

    # mot
    self.parser.add_argument('--data_cfg', type=str,
                             default='../src/lib/cfg/data.json',
                             help='load data from cfg')
    self.parser.add_argument('--data_dir', type=str, default='/data/yfzhang/MOT/JDE')

    # loss
    self.parser.add_argument('--mse_loss', action='store_true',
                             help='use mse loss or focal loss to train '
                                  'keypoint heatmaps.')

    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    self.parser.add_argument('--id_loss', default='ce',
                             help='reid loss: ce | triplet')
    self.parser.add_argument('--id_weight', type=float, default=1,
                             help='loss weight for id')
    self.parser.add_argument('--reid_dim', type=int, default=512,
                             help='feature dim for reid')

    self.parser.add_argument('--norm_wh', action='store_true',
                            help='norm_wh') 
    self.parser.add_argument('--dense_wh', action='store_true',
                             help='apply weighted regression near center or '
                                  'just apply regression on center point.')
    self.parser.add_argument('--cat_spec_wh', action='store_true',
                             help='category specific bounding box size.')
    self.parser.add_argument('--not_reg_offset', action='store_true',
                             help='not regress local offset.')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.pad = 31
    opt.num_stacks = 1
    opt.img_size = (1088, 608)
    return opt

  

  def init(self, args=''):
    default_dataset_info = {
      'mot': {'default_resolution': [608, 1088], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'jde', 'nID': 14455},
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    return opt
