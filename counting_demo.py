from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import dataloader as datasets
import torch
import cv2
from tracking_utils.timer import Timer
from tracking_utils import visualization as vis
from PIL import Image
from utils.bb_polygon import load_zone_anno
import numpy as np
import copy
from tracker.basetrack import BaseTrack

def eval_seq(opt, dataloader,polygon, paths, data_type, result_filename, frame_dir=None,save_dir=None,bbox_dir=None, show_image=True, frame_rate=30,polygon2=None,line1=None,line2=None,cam_id=None):
    count=0
    if save_dir:
        mkdir_if_missing(save_dir)
    if bbox_dir:
        mkdir_if_missing(bbox_dir)
    if frame_dir:
        mkdir_if_missing(frame_dir)
    if cam_id is not None:
        if cam_id is not None:
            f = open('/data/submission_output/cam_'+str(cam_id)+".txt", "w")
        else:
            f=None
    tracker = JDETracker(opt,polygon, paths, frame_rate=frame_rate,polygon2=polygon2)
    timer = Timer()
    results = []
    frame_id = 1

    for path, img, img0 in dataloader:
        img0_clone=copy.copy(img0)
        # if frame_id % 1 == 0:
        #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
           

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0) if opt.gpus[0]>=0 else torch.from_numpy(img).cpu().unsqueeze(0)
        online_targets,detection_boxes,out_of_polygon_tracklet = tracker.update(blob, img0)
        if f is not None:
            for frame_ind,_,track_type,mov_id in out_of_polygon_tracklet:
                if mov_id !='undetermine' and track_type !='undetermine':
                    if track_type in ['person','motor','motorcycle','bicycle',"tricycle"]:
                        track_type=1
                        f.write('cam_'+str(cam_id)+','+str(frame_ind)+','+str(mov_id)+','+str(track_type)+'\n')
                    elif track_type in ['car','van']:
                        track_type=2
                        f.write('cam_'+str(cam_id)+','+str(frame_ind)+','+str(mov_id)+','+str(track_type)+'\n')
                    elif track_type in ['bus']:
                        track_type=3
                        f.write('cam_'+str(cam_id)+','+str(frame_ind)+','+str(mov_id)+','+str(track_type)+'\n')
                    elif track_type in ['truck']:
                        track_type=4
                        f.write('cam_'+str(cam_id)+','+str(frame_ind)+','+str(mov_id)+','+str(track_type)+'\n')


        online_tlwhs = []
        online_ids = []
        
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
           
        #bbox detection plot        
        box_tlbrs=[]
        box_scores=[]
        box_occlusions=[]
        types=[]
        img_bbox=img0.copy()
        for box in detection_boxes:
            tlbr=box.tlbr
            tlwh=box.tlwh
           
            box_tlbrs.append(tlbr)
            box_scores.append(box.score)
            box_occlusions.append('occ' if box.occlusion_status==True else 'non_occ')
            types.append(box.infer_type())

        timer.toc()
        # save results
        for track in out_of_polygon_tracklet:
            frame_idx,id,classes,movement=track
            results.append((opt.input_video.split('/')[-1][:-4],frame_idx , classes, movement))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time,out_track=out_of_polygon_tracklet)
            bbox_im=vis.plot_detections(img_bbox,box_tlbrs,scores=box_scores,box_occlusion=None,types=types)
        if show_image:
            cv2.polylines(online_im,[np.asarray(polygon)],True,(0,255,255))
            cv2.polylines(bbox_im,[np.asarray(polygon)],True,(0,255,255))
            cv2.polylines(img0_clone,[np.asarray(polygon)],True,(0,255,255))
            cv2.imshow('online_im', online_im)
            cv2.imshow('bbox_im',bbox_im)
        # if save_dir is not None:
        #     cv2.polylines(online_im,[np.asarray(polygon)],True,(0,255,255))
        #     cv2.polylines(bbox_im,[np.asarray(polygon)],True,(0,255,255))
        #     # cv2.polylines(bbox_im,[np.asarray(paths['3'])],True,(0,255,255))
        #     # cv2.polylines(bbox_im,[np.asarray(paths['4'])],True,(0,255,255))

        #     if polygon2 is not None:
        #         cv2.polylines(online_im,[np.asarray(polygon2)],True,(0,0,255))
        #         cv2.polylines(bbox_im,[np.asarray(polygon2)],True,(0,0,255))
        #     if line1 is not None and line2 is not None:
        #         cv2.polylines(online_im,[np.asarray(line1)],True,(134,128,255))
        #         cv2.polylines(online_im,[np.asarray(line2)],True,(134,128,255))

        #     #cv2.polylines(img0_clone,[np.asarray(polygon)],True,(0,255,255))
        #     cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        #     cv2.imwrite(os.path.join(bbox_dir, '{:05d}.jpg'.format(frame_id)), bbox_im)
        #     cv2.imwrite(os.path.join(frame_dir, '{:05d}.jpg'.format(frame_id)),img0_clone)

        frame_id += 1
        # if frame_id==150:
        #     BaseTrack._count=0
        #     return frame_id, timer.average_time, timer.calls
        
    # save results
    return frame_id, timer.average_time, timer.calls

def demo(opt,polygon1,polygon2,prepath=None,cam_id=None):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    polygon, paths=load_zone_anno(opt.input_meta)
    if prepath is not None:
        paths=prepath
    polygon=np.int32(polygon1)
    #line1,line2=[polygon[4],polygon[3]],[polygon[1],polygon[2]]
    polygon2,_= np.int32(polygon2),None
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate
   
    print(cam_id)
    frame_tracking_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame_tracking')
    bbox_dir  = None if opt.output_format == 'text' else osp.join(result_root, 'bbox_detection')
    frame_dir =  None if opt.output_format == 'text' else osp.join(result_root, 'frame_dir')
    
    eval_seq(opt, dataloader,polygon, paths, 'mot', result_filename, frame_dir=frame_dir,save_dir=frame_tracking_dir,bbox_dir=bbox_dir, show_image=False, 
             frame_rate=frame_rate,polygon2=polygon2,line1=None,line2=None,cam_id=cam_id)

    # if opt.output_format == 'video':
    #     output_video_path = osp.join(result_root, 'result.mp4')
    #     cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
    #     os.system(cmd_str)

if __name__ == '__main__':
    cam_ids=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
    for cam_id in cam_ids:
        opt = opts().init()
        opt.cam_id=cam_id
        opt.input_video='/data/test_data/cam_'+cam_id+'.mp4'
        opt.input_meta ='/data/test_data/cam_'+cam_id+'.json'
        opt.output_root='results/cam_'+str(cam_id)
        paths=None
        if opt.cam_id=='01':
            opt.conf_thres=0.15
            opt.det_thres=0.15
            opt.near_cam_thres=0.3
            opt.near_cam_big_veh_thres=0.35
            opt.big_veh_thres=0.2
            polygon1=[
                    [
                        1.032258064516129,
                        341.0967741935484
                    ],
                    [
                        120.2258064516129,
                        210.4516129032258
                    ],
                    [
                        1030.1290322580645,
                        250.16129032258067
                    ],
                    [
                        1278.4516129032259,
                        464.48387096774195
                    ],
                    [
                        0,
                        462
                    ]
                ]
            polygon2=[
                    [
                        1.032258064516129,
                        341.0967741935484
                    ],
                    [
                        120.2258064516129,
                        210.4516129032258
                    ],
                    [
                        1030.1290322580645,
                        250.16129032258067
                    ],
                    [
                        1278.4516129032259,
                        464.48387096774195
                    ],
                    [
                        0,
                        462
                    ]
                ]
            opt.compound_coef=3
            opt_glob=opt
            from tracker.multitrackercam1 import JDETracker
        elif opt.cam_id=='02':
            opt.conf_thres=0.18
            opt.det_thres=0.18
            opt.near_cam_thres=0.2
            opt.near_cam_big_veh_thres=0.35
            opt.big_veh_thres=0.2
            polygon1=[
                    [
                        1.032258064516129,
                        341.0967741935484
                    ],
                    [
                        120.2258064516129,
                        210.4516129032258
                    ],
                    [
                        1030.1290322580645,
                        250.16129032258067
                    ],
                    [
                        1278.4516129032259,
                        464.48387096774195
                    ],
                    [
                        0,
                        462
                    ]
                ]
            polygon2=[
                    [
                        1.032258064516129,
                        341.0967741935484
                    ],
                    [
                        120.2258064516129,
                        210.4516129032258
                    ],
                    [
                        1030.1290322580645,
                        250.16129032258067
                    ],
                    [
                        1278.4516129032259,
                        464.48387096774195
                    ],
                    [
                        0,
                        462
                    ]
                ]
            opt.detection_model='Efficient'
            opt.compound_coef=3
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam2 import JDETracker
        elif opt.cam_id=='03':
            opt.conf_thres=0.15
            opt.det_thres=0.15
            opt.near_cam_thres=0.15
            opt.near_cam_big_veh_thres=0.35
            opt.big_veh_thres=0.2
            polygon1=[
                    [
                        1.032258064516129,
                        341.0967741935484
                    ],
                    [
                        120.2258064516129,
                        210.4516129032258
                    ],
                    [
                        1030.1290322580645,
                        250.16129032258067
                    ],
                    [
                        1278.4516129032259,
                        464.48387096774195
                    ],
                    [
                        0,
                        462
                    ]
                ]
            polygon2=[
                    [
                        1.032258064516129,
                        341.0967741935484
                    ],
                    [
                        120.2258064516129,
                        210.4516129032258
                    ],
                    [
                        1030.1290322580645,
                        250.16129032258067
                    ],
                    [
                        1278.4516129032259,
                        464.48387096774195
                    ],
                    [
                        0,
                        462
                    ]
                ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt_glob=opt
            from tracker.multitrackercam3 import JDETracker
        elif opt.cam_id=='04':
            opt.conf_thres=0.37
            opt.det_thres=0.37
            
            polygon1=[
                    [
                        1.032258064516129,
                        566.9032258064516
                    ],
                    [
                        484.0967741935484,
                        170.93548387096774
                    ],
                    [
                        756.6774193548387,
                        164.48387096774195
                    ],
                    [
                        910,
                        719
                    ],
                    [
                        0.22580645161290347,
                        718.516129032258
                    ]
                ]
            polygon2=[
                    [
                        1.032258064516129,
                        566.9032258064516
                    ],
                    [
                        294.0967741935484,
                        220.93548387096774
                    ],
                    [
                        796.6774193548387,
                        224.48387096774195
                    ],
                    [
                        910,
                        600
                    ],
                    [
                        0.22580645161290347,
                        600
                    ]
                ]
            opt.detection_model='FasterRcnn'
            opt.min_img_size=420
            opt_glob=opt
            from tracker.multitrackercam4 import JDETracker
        elif opt.cam_id=='05':
            opt.conf_thres=0.1
            opt.det_thres=0.1
            opt.track_buffer=9
            polygon1=[
                    [
                        1.032258064516129,
                        566.9032258064516
                    ],
                    [
                        484.0967741935484,
                        170.93548387096774
                    ],
                    [
                        756.6774193548387,
                        164.48387096774195
                    ],
                    [
                        910,
                        719
                    ],
                    [
                        0.22580645161290347,
                        718.516129032258
                    ]
                ]
            polygon2=[
                    [
                        1.032258064516129,
                        426.9032258064516
                    ],
                    [
                        294.0967741935484,
                        285.93548387096774
                    ],
                    [
                        796.6774193548387,
                        284.48387096774195
                    ],
                    [
                        910,
                        430
                    ],
                    [
                        0.22580645161290347,
                        430
                    ]
                ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam5 import JDETracker
        elif opt.cam_id=='06':
            opt.conf_thres=0.15
            opt.det_thres=0.15
            
            polygon1=[
                    [
                        7.483870967741936,
                        441.90322580645164
                    ],
                    [
                        299.741935483871,
                        220.7741935483871
                    ],
                    [
                        939.3225806451613,
                        220.19354838709677
                    ],
                    [
                        1225.225806451613,
                        465.2903225806452
                    ]
                ]
            polygon2=[
                    [
                        7.483870967741936,
                        441.90322580645164
                    ],
                    [
                        299.741935483871,
                        220.7741935483871
                    ],
                    [
                        939.3225806451613,
                        220.19354838709677
                    ],
                    [
                        1225.225806451613,
                        465.2903225806452
                    ]
                ]
            opt.detection_model='Efficient'
            opt.compound_coef=3
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam6 import JDETracker
        elif opt.cam_id=='07':
            opt.conf_thres=0.25
            opt.det_thres=0.25
            
            polygon1=[
                    [
                        19.580645161290324,
                        433.83870967741933
                    ],
                    [
                        284.4193548387097,
                        219.32258064516128
                    ],
                    [
                        1018.7741935483871,
                        226.5806451612903
                    ],
                    [
                        1270.3870967741937,
                        452.38709677419354
                    ]
                ]
            polygon2=[
                    [
                        19.580645161290324,
                        433.83870967741933
                    ],
                    [
                        284.4193548387097,
                        219.32258064516128
                    ],
                    [
                        1018.7741935483871,
                        226.5806451612903
                    ],
                    [
                        1270.3870967741937,
                        452.38709677419354
                    ]
                ]
            opt.detection_model='Efficient'
            opt.compound_coef=3
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam7 import JDETracker
        elif opt.cam_id=='08':
            opt.conf_thres=0.08
            opt.det_thres=0.08
            
            polygon1=[
                    [
                        19.580645161290324,
                        433.83870967741933
                    ],
                    [
                        304.4193548387097,
                        219.32258064516128
                    ],
                    [
                        988.7741935483871,
                        226.5806451612903
                    ],
                    [
                        1270.3870967741937,
                        452.38709677419354
                    ]
                ]
            polygon2=[
                    [
                        19.580645161290324,
                        433.83870967741933
                    ],
                    [
                        304.4193548387097,
                        219.32258064516128
                    ],
                    [
                        998.7741935483871,
                        226.5806451612903
                    ],
                    [
                        1270.3870967741937,
                        452.38709677419354
                    ]
                ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam8 import JDETracker
        elif opt.cam_id=='09':
            opt.conf_thres=0.15
            opt.det_thres=0.15
            
            polygon1=[
                    [
                        238.1290322580645,
                        454.80645161290323
                    ],
                    [
                        690.5483870967741,
                        286.258064516129
                    ],
                    [
                        1157.483870967742,
                        307.2258064516129
                    ],
                    [
                        1260.7096774193549,
                        613.6774193548387
                    ]
                ]
            polygon2=[
                    [
                        238.1290322580645,
                        454.80645161290323
                    ],
                    [
                        690.5483870967741,
                        286.258064516129
                    ],
                    [
                        1157.483870967742,
                        307.2258064516129
                    ],
                    [
                        1260.7096774193549,
                        613.6774193548387
                    ]
                ]
            opt.detection_model='FasterRcnn'
            opt.compound_coef=3
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam9 import JDETracker
        elif opt.cam_id=='10':
            opt.conf_thres=0.15
            opt.det_thres=0.15
            
            polygon1=[
                    [
                        0.7142857142857144,
                        656.5714285714286
                    ],
                    [
                        312.1428571428571,
                        258.4761904761905
                    ],
                    [
                        984.5238095238095,
                        254.66666666666663
                    ],
                    [
                        1276.904761904762,
                        505.1428571428571
                    ],
                    [
                        1278.8095238095239,
                        660.3809523809523
                    ]
                ]
            polygon2=[
                    [
                        0.7142857142857144,
                        656.5714285714286
                    ],
                    [252,
            336
            ],
                    [
                        272.1428571428571,
                        213.4761904761905
                    ],
                    [300,
                    214
                    ],
                    [449,
            256
            ],
                    [
                        984.5238095238095,
                        254.66666666666663
                    ],
                    [
                        1276.904761904762,
                        505.1428571428571
                    ],
                    [
                        1278.8095238095239,
                        660.3809523809523
                    ]
                ]
            paths={'1': [(491, 244), (355, 678)], '2': [(1139, 674), (754, 236)], '3': [(455, 238), (193, 325)], '4': [(59, 532), (732, 241)], '5': [(936, 675), (171, 349)], '6': [(77, 499), (214, 697)], '7': [(998, 679), (1178, 394)], '8': [(55, 510), (1178, 388)], '9': [(1147, 365), (775, 243)], '10': [(726,256), (1139, 388)], '11': [(1164, 370), (179, 335)], '12': [(1165, 300), (375, 693)]}
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam10 import JDETracker
        elif opt.cam_id=='11':
            opt.conf_thres=0.25
            opt.det_thres=0.25
            
            polygon1=[
                    [
                        434.0967741935484,
                        369.4516129032258
                    ],
                    [
                        797.8064516129033,
                        369.6774193548387
                    ],
                    [
                        984.9032258064516,
                        574.1612903225806
                    ],
                    [
                        249.41935483870967,
                        584.6451612903226
                    ]
                ]
            polygon2=[
                    [
                        434.0967741935484,
                        360.4516129032258
                    ],
                    [
                        797.8064516129033,
                        363.6774193548387
                    ],
                    [
                        984.9032258064516,
                        574.1612903225806
                    ],
                    [
                        249.41935483870967,
                        584.6451612903226
                    ]
                ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam11 import JDETracker
        elif opt.cam_id=='12':
            opt.conf_thres=0.25
            opt.det_thres=0.25
            
            polygon1=[
                    [
                        0.04761904761904,
                        495.5238095238095
                    ],
                    [
                        480.2380952380952,
                        330.6190476190476
                    ],
                    [
                        1270.7142857142856,
                        435.85714285714283
                    ],
                    [
                        1280,
                        719
                    ],
            [
                        0.04761904761904,
                        635.5238095238095
                    ]

                ]
            polygon2=[
                    [
                        0.04761904761904,
                        495.5238095238095
                    ],
                    [
                        410.2380952380952,
                        353.6190476190476
                    ],
                    [
                        1272.7142857142856,
                        491.85714285714283
                    ],
                    [
                        1280,
                        719
                    ],
            [
                        0.04761904761904,
                        635.5238095238095
                    ]
                ]
            opt.detection_model='Efficient'
            opt.compound_coef=3
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam12 import JDETracker
        elif opt.cam_id=='13':
            opt.conf_thres=0.15
            opt.det_thres=0.15
            
            polygon1= [
                    [
                        9.096774193548388,
                        340.2903225806452
                    ],
                    [
                        343.7741935483871,
                        166.09677419354838
                    ],
                    [
                        1009.9032258064516,
                        171.74193548387098
                    ],
                    [
                        1276.032258064516,
                        314.48387096774195
                    ]
                ]
            polygon2=[
                    [
                        9.096774193548388,
                        340.2903225806452
                    ],
                    [
                        230.7741935483871,
                        211.09677419354838
                    ],
                    [
                        1079.9032258064516,
                        216.74193548387098
                    ],
                    [
                        1286.032258064516,
                        314.48387096774195
                    ]
                ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam13 import JDETracker
        elif opt.cam_id=='14':
            opt.conf_thres=0.15
            opt.det_thres=0.15
            
            polygon1= [
                [
                    11.516129032258064,
                    466.0967741935484
                ],
                [
                    305.06451612903226,
                    216.9032258064516
                ],
                [
                    951.8387096774194,
                    182.2258064516129
                ],
                [
                    1272.8064516129032,
                    416.0967741935484
                ]
            ]
            polygon2=[
                [
                    11.516129032258064,
                    466.0967741935484
                ],
                [
                    305.06451612903226,
                    216.9032258064516
                ],
                [
                    951.8387096774194,
                    182.2258064516129
                ],
                [
                    1272.8064516129032,
                    416.0967741935484
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=3
            opt.min_img_size=400
            opt_glob=opt
            paths={'1': [(424, 201), (224, 485)], '2': [(827, 508), (628, 178)], '3': [(1175, 200), (644, 175)], '4': [(1175, 304), (205, 519)], '5': [(1101, 554), (1179, 277)], '6': [(414, 195), (1165, 266)]}
            from tracker.multitrackercam14 import JDETracker
        elif opt.cam_id=='15':
            opt.conf_thres=0.15
            opt.det_thres=0.15
            
            polygon1= [
                [
                    11.516129032258064,
                    466.0967741935484
                ],
                [
                    305.06451612903226,
                    216.9032258064516
                ],
                [
                    951.8387096774194,
                    182.2258064516129
                ],
                [
                    1272.8064516129032,
                    416.0967741935484
                ]
            ]
            polygon2=[
                [
                    11.516129032258064,
                    466.0967741935484
                ],
                [
                    305.06451612903226,
                    216.9032258064516
                ],
                [
                    951.8387096774194,
                    182.2258064516129
                ],
                [
                    1272.8064516129032,
                    416.0967741935484
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=3
            opt.min_img_size=400
            opt_glob=opt
            paths={'1': [(424, 201), (224, 485)], '2': [(827, 508), (628, 178)], '3': [(1175, 200), (644, 175)], '4': [(1175, 304), (205, 519)], '5': [(1101, 554), (1179, 277)], '6': [(414, 195), (1165, 266)]}
            from tracker.multitrackercam15 import JDETracker
        elif opt.cam_id=='16':
            opt.conf_thres=0.11
            opt.det_thres=0.11
            
            polygon1= [
                [
                    129.25806451612902,
                    637.8709677419355
                ],
                [
                    97.806451612903224,
                    190.09677419354838
                ],
                [
                    501.0,
                    152.5483870967742
                ],
		[
                    679.0,
                    238.5483870967742
                ]
                 ,
                [
                   920,
                   251
                ]
		,
                [
                    1279.2903225806451,
                    442.2258064516129
                ]
            ]
            polygon2=[
                [
                    129.25806451612902,
                    637.8709677419355
                ],
                [
                    97.806451612903224,
                    190.09677419354838
                ],
                [
                    501.0,
                    152.5483870967742
                ],
		[
                    679.0,
                    238.5483870967742
                ]
                 ,
                [
                   920,
                   251
                ]
		,
                [
                    1279.2903225806451,
                    442.2258064516129
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=3
            opt.min_img_size=400
            opt_glob=opt
            paths={'1': [(155, 166), (596, 621)], '2': [(221, 162), (679, 209)], '3': [(909, 287), (878, 619)]}
            from tracker.multitrackercam16 import JDETracker
        elif opt.cam_id=='17':
            opt.conf_thres=0.12
            opt.det_thres=0.12
            
            polygon1= [
                [
                    129.25806451612902,
                    637.8709677419355
                ],
                [
                    97.806451612903224,
                    190.09677419354838
                ],
                [
                    501.0,
                    152.5483870967742
                ],
		[
                    679.0,
                    238.5483870967742
                ]
                 ,
                [
                   920,
                   251
                ]
		,
                [
                    1279.2903225806451,
                    442.2258064516129
                ]
            ]
            polygon2=[
                [
                    129.25806451612902,
                    637.8709677419355
                ],
                [
                    97.806451612903224,
                    190.09677419354838
                ],
                [
                    501.0,
                    152.5483870967742
                ],
		[
                    679.0,
                    238.5483870967742
                ]
                 ,
                [
                   920,
                   251
                ]
		,
                [
                    1279.2903225806451,
                    442.2258064516129
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            {'1': [(155, 166), (596, 621)], '2': [(221, 162), (679, 209)], '3': [(909, 287), (878, 619)]}
            from tracker.multitrackercam17 import JDETracker
        elif opt.cam_id=='18':
            opt.conf_thres=0.35
            opt.det_thres=0.35
            
            polygon1= [
                [
                    131.67741935483872,
                    307.2258064516129
                ],
                [
                    605.0645161290323,
                    148.3548387096774
                ],
                [
                    941.3548387096774,
                    249.9677419354839
                ],
                [
                    885.7096774193549,
                    691.0967741935484
                ]
            ]
            polygon2=[
                [
                    131.67741935483872,
                    307.2258064516129
                ],
                [
                    605.0645161290323,
                    148.3548387096774
                ],
                [
                    941.3548387096774,
                    249.9677419354839
                ],
                [
                    885.7096774193549,
                    691.0967741935484
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam18 import JDETracker
        elif opt.cam_id=='19':
            opt.conf_thres=0.35
            opt.det_thres=0.35
            
            polygon1= [
                [
                    76.67741935483872,
                    260.2258064516129
                ],
                [
                    528.0645161290323,
                    111.3548387096774
                ],
                [
                    941.3548387096774,
                    249.9677419354839
                ],
                [
                    885.7096774193549,
                    691.0967741935484
                ]
            ]
            polygon2=[
                [
                    76.67741935483872,
                    260.2258064516129
                ],
                [
                    528.0645161290323,
                    111.3548387096774
                ],
                [
                    941.3548387096774,
                    249.9677419354839
                ],
                [
                    885.7096774193549,
                    691.0967741935484
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam19 import JDETracker
        elif opt.cam_id=='20':
            opt.conf_thres=0.17
            opt.det_thres=0.17
            
            polygon1= [
                [
                    74.41935483870968,
                    212.8709677419355
                ],
                [
                    759.9032258064516,
                    146.74193548387098
                ],
                [
                    1241.3548387096773,
                    293.51612903225805
                ],
                [
                    534.0967741935484,
                    686.258064516129
                ]
            ]
            polygon2=[
                [
                    74.41935483870968,
                    212.8709677419355
                ],
                [
                    759.9032258064516,
                    146.74193548387098
                ],
                [
                    1241.3548387096773,
                    293.51612903225805
                ],
                [
                    534.0967741935484,
                    686.258064516129
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            paths={'1': [(336, 173), (1149, 515)], '2': [(1230, 380), (498, 147)], '3': [(1226, 396), (51, 384)], '4': [(356, 170), (62, 405)], '5': [(121, 424), (485, 151)], '6': [(127, 438), (1184, 448)]}
            from tracker.multitrackercam20 import JDETracker
        elif opt.cam_id=='21':
            opt.conf_thres=0.15
            opt.det_thres=0.15
            
            polygon1= [
                [
                    5.064516129032258,
                    388.6774193548387
                ],
                [
                    439.741935483871,
                    211.25806451612902
                ],
                [
                    813.9354838709678,
                    252.38709677419354
                ],
                [
                    830.0645161290323,
                    661.258064516129
                ]
            ]
            polygon2=[
                [
                    5.064516129032258,
                    388.6774193548387
                ],
                [
                    439.741935483871,
                    211.25806451612902
                ],
                [
                    813.9354838709678,
                    252.38709677419354
                ],
                [
                    830.0645161290323,
                    661.258064516129
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=3
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam21 import JDETracker
        elif opt.cam_id=='22':
            opt.conf_thres=0.1
            opt.det_thres=0.1
            
            polygon1= [
                [
                    5.064516129032258,
                    388.6774193548387
                ],
                [
                    439.741935483871,
                    211.25806451612902
                ],
                [
                    813.9354838709678,
                    252.38709677419354
                ],
                [
                    830.0645161290323,
                    661.258064516129
                ]
            ]
            polygon2=[
                [
                    5.064516129032258,
                    388.6774193548387
                ],
                [
                    439.741935483871,
                    211.25806451612902
                ],
                [
                    813.9354838709678,
                    252.38709677419354
                ],
                [
                    830.0645161290323,
                    661.258064516129
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam22 import JDETracker
        elif opt.cam_id=='23':
            opt.conf_thres=0.28
            opt.det_thres=0.28
            
            polygon1=[
                [
                    207.48387096774195,
                    417.7096774193548
                ],
                [
                    803.4516129032259,
                    283.03225806451616
                ],
                [
                    1153.4516129032259,
                    367.7096774193548
                ],
                [
                    1089.741935483871,
                    690.2903225806451
                ]
            ] 
            polygon2=[
                [
                    247.48387096774195,
                    428.7096774193548
                ],
                [
                    803.4516129032259,
                    283.03225806451616
                ],
                [
                    1153.4516129032259,
                    367.7096774193548
                ],
                [
                    1089.741935483871,
                    690.2903225806451
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=3
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam23 import JDETracker
        elif opt.cam_id=='24':
            opt.conf_thres=0.18
            opt.det_thres=0.18
            
            polygon1= [
                [
                    207.48387096774195,
                    417.7096774193548
                ],
                [
                    803.4516129032259,
                    283.03225806451616
                ],
                [
                    1153.4516129032259,
                    367.7096774193548
                ],
                [
                    1089.741935483871,
                    690.2903225806451
                ]
            ]
            polygon2=[
                [
                    207.48387096774195,
                    417.7096774193548
                ],
                [
                    803.4516129032259,
                    283.03225806451616
                ],
                [
                    1153.4516129032259,
                    367.7096774193548
                ],
                [
                    1089.741935483871,
                    690.2903225806451
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam24 import JDETracker
        elif opt.cam_id=='25':
            opt.conf_thres=0.12
            opt.det_thres=0.12
            
            polygon1= [
                [
                    207.48387096774195,
                    417.7096774193548
                ],
                [
                    803.4516129032259,
                    283.03225806451616
                ],
                [
                    1153.4516129032259,
                    367.7096774193548
                ],
                [
                    1089.741935483871,
                    690.2903225806451
                ]
            ]
            polygon2=[
                [
                    207.48387096774195,
                    417.7096774193548
                ],
                [
                    803.4516129032259,
                    283.03225806451616
                ],
                [
                    1153.4516129032259,
                    367.7096774193548
                ],
                [
                    1089.741935483871,
                    690.2903225806451
                ]
            ]
            opt.detection_model='Efficient'
            opt.compound_coef=4
            opt.min_img_size=400
            opt_glob=opt
            from tracker.multitrackercam25 import JDETracker        
    
        
        demo(opt,polygon1,polygon2,paths,cam_id)

