from collections import deque
import  copy
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append('/home/lam/HCMAIChallenge')
from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from tracker.basetrack import BaseTrack, TrackState
from scipy.spatial.distance import cdist
from imutils.object_detection import non_max_suppression
import math
from torchvision.transforms import Resize,Normalize,ToTensor,Compose
from PIL import Image
from functools import reduce
import yaml
from EfficientDet.backbone import EfficientDetBackbone
from EfficientDet.efficientdet.utils import BBoxTransform, ClipBoxes
from EfficientDet.utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import cv2
from utils.bb_polygon import check_bbox_intersect_or_outside_polygon,check_bbox_outside_polygon,counting_moi,point_to_line_distance,check_bbox_inside_polygon,tlbrs_to_mean_area,box_line_relative

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    out_of_frame_patience=5
    num_cluster=5
    type_infer_patience=4
    score_infer_type_thres=0.6
    def __init__(self, tlwh, score, vehicle_type, buffer_size=30,temp_feat=None,huge_vehicle=False):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        
        self.tracklet_len = 0

        self.smooth_feat = None
        # self.update_features(temp_feat,None)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.6
        self.num_out_frame=0
        self.cluster_features={'centers':[],'cluster':[]}
        self.track_frames=[]
        self.w_hs=[]
        self.occlusion_status=False # use for bbox only
        self.iou_box=None #use for bbox only
        self.box_hist=[]
        self.vehicle_types_list=[]
        self.vehicle_types_list.append(vehicle_type)
        self.track_trajectory=[]
        self.track_trajectory.append(self.tlwh_to_tlbr(tlwh))
        self.huge_vehicle=huge_vehicle
    def update_cluster(self,feat):
        feat /= np.linalg.norm(feat)
        if len(self.cluster_features['cluster'])<STrack.num_cluster:
            self.cluster_features['cluster'].append([feat])
            self.cluster_features['centers'].append(feat)
        else:
            min_center=np.argmin(np.squeeze(cdist(self.cluster_features['centers'], [feat], metric='cosine')))
            self.cluster_features['cluster'][min_center].append(feat)
            self.cluster_features['centers'][min_center]=np.mean(self.cluster_features['cluster'][min_center],axis=0)
            self.cluster_features['centers']/=np.linalg.norm( self.cluster_features['centers'])
    
        
    def update_features(self, feat,iou_box):
        #feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if iou_box==None: iou_box=0
            update_param=(1-self.alpha)*iou_box+self.alpha
            #self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat = update_param * self.smooth_feat + (1 -update_param) * feat
            self.box_hist.append(update_param)
            
                               
        self.features.append(feat)
        #self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        # self.track_trajectory.append(self.tlbr)
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.track_frames.append(frame_id)

    def re_activate(self, new_track, frame_id, new_id=False):
        new_tlwh = new_track.tlwh
        self.track_trajectory.append(self.tlwh_to_tlbr(new_tlwh))
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        #self.update_features(new_track.curr_feat,new_track.iou_box)
        #self.update_cluster(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.track_frames.append(frame_id)

    def update(self, new_track, frame_id, update_feature=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.vehicle_types_list.append(new_track.vehicle_types_list[-1])

        new_tlwh = new_track.tlwh
        self.track_trajectory.append(self.tlwh_to_tlbr(new_tlwh))

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat,new_track.iou_box) ###########
            #self.update_cluster(new_track.curr_feat)
        self.track_frames.append(frame_id)
    def infer_type(self):
        def most_frequent(List): 
            return max(set(List), key = List.count)
        types=most_frequent(self.vehicle_types_list)
        return types
        # if classes in ['bicycle', 'motorcycle']:
        #     return 1
        # elif classes =='car':
        #     return 2
        # elif classes=='bus':
        #     return 3
        # else:
        #     return 4
    @property
    def vehicle_type(self):
        def most_frequent(List): 
            return max(set(List), key = List.count)
        if len(self.track_frames)>=self.type_infer_patience:
            return most_frequent(self.vehicle_types_list)
        elif self.score >=self.score_infer_type_thres:
            return most_frequent(self.vehicle_types_list)
        else:
            return 'Undetermine'

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt,polygon, paths, polygon2=None,frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.input_size = input_sizes[opt.compound_coef] 
        self.obj_list =['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']
        self.person_or_motorcycle=['person']
        self.obj_interest=[ 'motorcycle','bicycle', 'bus', 'truck','car'] if self.person_or_motorcycle[0]!='person' else [ 'person', 'bus', 'truck','car']
        print(self.obj_interest)
        self.detetection_model= EfficientDetBackbone(compound_coef=opt.compound_coef, num_classes=len(self.obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
       
        self.detetection_model.load_state_dict(torch.load(f'EfficientDet/weights/efficientdet-d{opt.compound_coef}.pth'))
        self.detetection_model.eval()
        device = torch.device('cuda:0')
        self.detetection_model = self.detetection_model.to(device)

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K

        self.kalman_filter = KalmanFilter()

        self.polygon=polygon
        self.paths=paths
        self.polygon2=polygon2
        
        self.line1=[self.polygon[1],self.polygon[2]]
        self.two_polygon_system=True
        self.warmup_frame=6 if self.two_polygon_system else 0
        self.virtual_polygon= np.int32([
                [
                    0,
                    720
                ],
                [
                    0,
                    209
                ],
                [
                    1270,
                    209
                ],
                [
                    1270,
                    720
                ]
            ])

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        init_polygon=self.polygon2 if self.two_polygon_system and self.frame_id>= self.warmup_frame else self.polygon
        two_wheel_polygon=init_polygon
        four_wheel_polygon=self.polygon
        virtual_polygon=self.virtual_polygon
        huge_box_thres=230
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            ori_imgs, framed_imgs, framed_metas = preprocess([img0], max_size=self.input_size)
            device = torch.device('cuda:0')
            x = torch.stack([torch.from_numpy(fi).to(device) for fi in framed_imgs], 0)
            x = x.to(torch.float32 ).permute(0, 3, 1, 2)
            features, regression, classification, anchors = self.detetection_model(x)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        self.opt.det_thres, self.opt.nms_thres)
            out = invert_affine(framed_metas, out)
            bbox=[]
            score=[]
            types=[]
            huge_vehicles=[]
            for j in range(len(out[0]['rois'])):
                obj = self.obj_list[out[0]['class_ids'][j]]
                if obj in self.obj_interest:
                    x1, y1, x2, y2 = out[0]['rois'][j].astype(np.int)
                    #bike,bicycle
                    if (y1+y2)/2>0.72*height and float(out[0]['scores'][j])<=0.3:
                        continue
                   
                    if obj not in self.person_or_motorcycle and float(out[0]['scores'][j])>=0.2:
                        bbox.append([x1, y1, x2, y2])
                        score.append( float(out[0]['scores'][j]))
                        types.append(obj)
                        huge_vehicles.append(False if (y2-y1)<=huge_box_thres else True )
                    elif obj in self.person_or_motorcycle: #['bicycle',  'motorcycle']
                        bbox.append([x1, y1, x2, y2])
                        score.append( float(out[0]['scores'][j]))
                        types.append(obj)
                        huge_vehicles.append(False)
            

        # vis
        # print(len(bbox))
        # print(img0.shape)
        # print(self.polygon)
        # for i in range(len(bbox)):
        #     bb = bbox[i]
        #     cv2.rectangle(img0, (bb[0], bb[1]),
        #                   (bb[2], bb[3]),
        #                   (0, 255, 0), 2)
        # cv2.polylines(img0,[np.asarray(self.polygon)],True,(0,255,255))
        # cv2.imshow('dets', img0)
        # cv2.waitKey(0)
        
        

        if len(bbox) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), sco, clas, 30,huge_vehicle=hv) for
                          (tlbr, sco,clas,hv) in zip(bbox,score,types,huge_vehicles)]
            
        else:
            detections = []
        
        detections_plot=copy.deepcopy(detections)


        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with gating distance'''
        strack_pool,lost_map_tracks = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        #dists = matching.embedding_distance(strack_pool, detections)
        detections=heuristic_occlusion_detection(detections)
        match_thres=100
        dists=np.zeros(shape=(len(strack_pool),len(detections)))
        dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections,type_diff=True)
        #dists = matching.fuse_motion(self.opt,self.kalman_filter, dists, strack_pool, detections,lost_map=lost_map_tracks,occlusion_map=occlusion_map,thres=match_thres)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=match_thres)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        ''' '''
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.6)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.huge_vehicle:
                track_init_polygon=virtual_polygon
            elif track.infer_type() in ['bus','truck','car']:
                track_init_polygon=self.polygon
            else:
                track_init_polygon=init_polygon
            if track.score < self.det_thresh or track.occlusion_status==True or  check_bbox_outside_polygon(track_init_polygon,track.tlbr):
                continue
            # track_types=self.person_or_motorcycle[0] if tlbrs_to_mean_area(track.track_trajectory) <=1500 else track.infer_type()
            if self.frame_id>=5  and not check_bbox_inside_polygon(track_init_polygon,track.tlbr):#and track_types in self.person_or_motorcycle #person, motorcycle
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state and getting out of interest tracklet if have"""
        out_of_polygon_tracklet=[]
        refind_stracks_copy=[]
        activated_starcks_copy=[]
        for idx,current_tracked_tracks in enumerate([refind_stracks,activated_starcks]) :#
        
            for track in current_tracked_tracks:
                if tlbrs_to_mean_area(track.track_trajectory) <=1000 :
                    track_type= self.person_or_motorcycle[0] #person
                else:
                    track_type=track.infer_type()
                if track_type in self.person_or_motorcycle:
                    out_polygon=two_wheel_polygon
                    p_type='two_wheel'
                else:
                    out_polygon=four_wheel_polygon #if not track.huge_vehicle else virtual_polygon
                    p_type='four_wheel'
                if check_bbox_outside_polygon(out_polygon,track.tlbr) :
                    track.mark_removed()
                    removed_stracks.append(track)
                    if ((len(track.track_frames)>=2 and self.frame_id <=5) or (len(track.track_frames)>=5 and self.frame_id>=self.warmup_frame+5)) and idx==1:########## 4 is confident number of frame

                        track_center=[ [(x[0]+x[2])/2,(x[1]+x[3])/2] for x in track.track_trajectory]
                        movement_id=counting_moi(self.paths,[(track_center[0],track_center[-1])])[0]
                        line_interest=self.line1 
                        out_direction='up'
                        frame_id=self.frame_id+kalman_predict_out_line(track,line_interest,out_direction)
                        out_of_polygon_tracklet.append((frame_id,track.track_id,track_type,movement_id))
                else:
                    refind_stracks_copy.append(track) if idx ==0 else activated_starcks_copy.append(track)
        refind_stracks=refind_stracks_copy
        activated_starcks=activated_starcks_copy
        

        lost_stracks_copy=[]
        for track in lost_stracks:
            if tlbrs_to_mean_area(track.track_trajectory) <=1000 :
                track_type= self.person_or_motorcycle[0] #person
            else:
                track_type=track.infer_type()
            if track_type in self.person_or_motorcycle:
                out_polygon=two_wheel_polygon
                p_type='two_wheel'
            else:
                out_polygon=four_wheel_polygon if not track.huge_vehicle else virtual_polygon
                p_type='four_wheel'
           
            if check_bbox_intersect_or_outside_polygon(out_polygon,track.tlbr) :
                track.mark_removed()
                removed_stracks.append(track)
                if ((len(track.track_frames)>=2 and self.frame_id <=5) or (len(track.track_frames)>=6 and self.frame_id>=self.warmup_frame+5)):
                    track_center=[ [(x[0]+x[2])/2,(x[1]+x[3])/2] for x in track.track_trajectory]
                    movement_id=counting_moi(self.paths,[(track_center[0],track_center[-1])])[0]
                    line_interest=self.line1 
                    out_direction='up'
                    frame_id=self.frame_id+kalman_predict_out_line(track,line_interest,out_direction)
                    out_of_polygon_tracklet.append((frame_id,track.track_id,track_type,movement_id))
            else:
                lost_stracks_copy.append(track)

        lost_stracks=lost_stracks_copy









        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost :
                track.mark_removed()
                removed_stracks.append(track)
            #Remove out of screen tracklet
            elif track.tlwh[0]+track.tlwh[2]//2>width or track.tlwh[1]+track.tlwh[3]//2>height or min(track.tlwh[0]+track.tlwh[2]//2,track.tlwh[1]+track.tlwh[3]//2)<0:
                track.num_out_frame+=1
                if track.num_out_frame>STrack.out_of_frame_patience:
                    track.mark_removed()
                    removed_stracks.append(track)
       
        # print('Ramained match {} s'.format(t4-t3))
        # print(out_of_polygon_tracklet)
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks,_ = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks,_ = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        #self.merge_track()
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        
        return output_stracks,detections_plot,out_of_polygon_tracklet

    #can paralel,current bottleneck of model
    def merge_track(self,min_thres=0.2,distance_thres=15, consitence_thres=10):
        def is_overlap(lost_track,tracked_track):
            if tracked_track.start_frame>lost_track.end_frame or lost_track.start_frame>tracked_track.end_frame:
                return False
        def predict_future(lost_track,num_frame):
            mean,cov=lost_track.mean,lost_track.covariance
            for _ in range(num_frame):
                mean,cov=lost_track.kalman_filter.predict(mean,cov)
            return mean,cov
        def cluster_compare(lost_track,tracked_track):
            return np.min(cdist(lost_track.cluster_features['centers'],tracked_track.cluster_features['centers'],metric='cosine'))
            
        def distance(lost_track,tracked_track,min_thres=min_thres,distance_thres=distance_thres):
            if is_overlap(lost_track,tracked_track):
                return np.inf
            else:
                pred_mean,pred_cov=predict_future(lost_track,tracked_track.start_frame-lost_track.end_frame)
               
                tracked_xyah=STrack.tlwh_to_xyah(tracked_track._tlwh)
                if self.kalman_filter.gating_distance(pred_mean,pred_cov,tracked_xyah) > distance_thres:
                    return np.inf
                else:
                    return cluster_compare(lost_track,tracked_track)
        
        cost_matrix=np.zeros(shape=(len(self.lost_stracks),len(self.tracked_stracks)))
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if min(len(self.lost_stracks[i].track_frames),len(self.tracked_stracks[j].track_frames))<=consitence_thres:
                    cost_matrix[i][j]=np.inf
                else:
                    cost_matrix[i][j]=distance(self.lost_stracks[i],self.tracked_stracks[j])
        
        matches,_,_=matching.linear_assignment(cost_matrix,thresh=min_thres)
        map_lost_track=np.ones_like(self.lost_stracks,dtype=np.int)

        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i][j]<=1:
                    print('sim of ' +str(self.lost_stracks[i].track_id) + ' and ' +str(self.tracked_stracks[j].track_id) +' : '+str(cost_matrix[i][j]) )
        if len(matches)==0:
            return
      
        for ilost_track,i_tracked_track in matches:
            print('------------------------------------')
            print('merge ' + str(self.tracked_stracks[i_tracked_track].track_id)+' to '+str(self.lost_stracks[ilost_track].track_id))
            map_lost_track[ilost_track]=0 # pylint: disable=unsupported-assignment-operation
            for num_clus in range(len(self.tracked_stracks[i_tracked_track].cluster_features['cluster'])):
                for clus in self.tracked_stracks[i_tracked_track].cluster_features['cluster'][num_clus]:
                    self.lost_stracks[ilost_track].update_cluster(clus)
            self.lost_stracks[ilost_track].mean,self.lost_stracks[ilost_track].covariance=self.tracked_stracks[i_tracked_track].mean,self.tracked_stracks[i_tracked_track].covariance
            self.lost_stracks[ilost_track].track_frames+=self.tracked_stracks[i_tracked_track].track_frames
            self.lost_stracks[ilost_track].frame_id=self.tracked_stracks[i_tracked_track].frame_id
            self.tracked_stracks[i_tracked_track]=self.lost_stracks[ilost_track]
        
        new_lost_tracks=[]
        for ilost_track in range(len(map_lost_track)):
            if map_lost_track[ilost_track] ==1:
                new_lost_tracks.append(self.lost_stracks[ilost_track])
        self.lost_stracks=new_lost_tracks

def kalman_predict_out_line(track,line,out_direction):
    # print(track.track_id)
    # print(line)
    # print(out_direction)
    # print(track.tlbr)
    if box_line_relative(track.tlbr,line)==out_direction:
        return 0
    predict_num_out=0
    prev_mean,prev_cov=track.mean,track.covariance
    kal_man=KalmanFilter()
    predict_thres=5 if out_direction=='up' else 0
    max_long_predict=20 if out_direction=='up' else 2 if track.infer_type() in ['person','motorcycle','biycycle'] else 5
    while  box_line_relative(mean_to_tlbr(prev_mean),line) !=out_direction:
        predict_num_out+=1
        cur_mean=prev_mean #of t
        mean,cov=kal_man.predict(prev_mean,prev_cov)
        if predict_num_out>predict_thres:
            new_mean,new_cov=mean,cov
        else:
            new_mean,new_cov= kal_man.update(prev_mean,prev_cov,mean[:4])
        prev_mean,prev_cov=new_mean,new_cov #of t+1
        if predict_num_out>=max_long_predict or np.sum(np.abs(cur_mean-mean))==0:
            break
        # print(mean_to_tlbr(mean))

    return predict_num_out
import copy
def mean_to_tlbr(mean):
    ret = copy.copy(mean[:4])
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    ret[2:] += ret[:2]
    return ret
    


def heuristic_occlusion_detection(detections,thres=0.5): #0.5
    detection_tlbrscores=  [np.append(detection.tlbr,[detection.score]) for detection in detections] 
    detection_tlbrscores=  np.asarray(detection_tlbrscores)
    occ_iou=[]
    new_detection_pool=[]
    for idx,detection_tlbrscore in enumerate(detection_tlbrscores):
        xA=np.maximum([detection_tlbrscore[0]]*len(detections),detection_tlbrscores[:,0])
        yA=np.maximum([detection_tlbrscore[1]]*len(detections),detection_tlbrscores[:,1])
        xB=np.minimum([detection_tlbrscore[2]]*len(detections),detection_tlbrscores[:,2])
        yB=np.minimum([detection_tlbrscore[3]]*len(detections),detection_tlbrscores[:,3])
        interArea = np.maximum(0, xB - xA ) * np.maximum(0, yB - yA )
        box_area=(detection_tlbrscore[2]-detection_tlbrscore[0])*(detection_tlbrscore[3]-detection_tlbrscore[1])
        box_ious= np.asarray(interArea/box_area)

        delta_scores=np.asarray(detection_tlbrscores[:,4]-detection_tlbrscore[4])
        num_invalid= len( np.where(np.logical_and(box_ious>thres, delta_scores >=0))[0]) #-0.03
        num_invalid_thres2=len(np.where(box_ious>0.85)[0]) #0.55

        detections[idx].iou_box=np.sort(box_ious)[-2] if len(box_ious)>1 else None
        occ_iou.append(detections[idx].iou_box)
        if num_invalid >=2 :
            detections[idx].occlusion_status=True
            
        else:
            new_detection_pool.append(detections[idx])
        if (num_invalid_thres2>=2 and box_area>40000):
            detections[idx].vehicle_types_list[-1]='truck'
    
    return new_detection_pool







def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    lost_map=[]
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
        lost_map.append(0)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
            lost_map.append(1)
    return res,lost_map


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb