import lap
import numpy as np
import scipy
from cython_bbox import bbox_overlaps as bbox_ious #pylint: disable=no-name-in-module
from scipy.spatial.distance import cdist
import sys
sys.path.append('/home/lam/HCMAIChallenge')
from tracking_utils import kalman_filter
def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    occlusion_map=[btrack.occlusion_status for btrack in btracks]
    cost_matrix = (1 - _ious) *(np.asarray(occlusion_map,dtype=np.int)*1000+1) if len(occlusion_map)>0 else 1-_ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarrays
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    #normalize to (0,1)
    min_cost_matrix=0.002#0.005
    max_cost_matrix=0.25#0.295
    print(np.max(cost_matrix))
    print(np.min(cost_matrix))
    cost_matrix=(cost_matrix-min_cost_matrix)/(max_cost_matrix-min_cost_matrix)
    return cost_matrix

#bottleneck here
def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False,type_diff=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]*85         #cam9: 7, cam4: 85
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row]=gating_distance
    
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    if type_diff:
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[i])):
                if tracks[i].vehicle_type!='Undetermine' and tracks[i].vehicle_type!=detections[j].infer_type() and tracks[i].vehicle_type in ['bus','truck']:
                    cost_matrix[i][j]=np.inf
           
    return cost_matrix

def gate_cost_matrix4(kf, cost_matrix, tracks, detections, only_position=False,type_diff=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]*85         #cam9: 7, cam4: 85
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row]=gating_distance
    
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    if type_diff:
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[i])):
                if tracks[i].vehicle_type!='Undetermine' and tracks[i].vehicle_type!=detections[j].infer_type() :
                    if tracks[i].vehicle_type in ['motor','bicycle','person','car']:
                        cost_matrix[i][j]=np.inf
                    elif detections[j].infer_type() not in ['car','bus','truck']:
                        cost_matrix[i][j]=np.inf
            
           
    return cost_matrix
def gate_cost_matrix6(kf, cost_matrix, tracks, detections, only_position=False,type_diff=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]*85         #cam9: 7, cam4: 85
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row]=gating_distance
    
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    if type_diff:
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[i])):
                if tracks[i].vehicle_type!='Undetermine' and tracks[i].vehicle_type!=detections[j].infer_type() :
                    if tracks[i].vehicle_type in ['motor','bicycle','person','car'] and detections[j].infer_type() in ['bus','truck']:
                        cost_matrix[i][j]=np.inf
                    elif detections[j].infer_type() not in ['car','bus','truck']:
                        cost_matrix[i][j]=np.inf
            
           
    return cost_matrix
def gate_cost_matrix5(kf, cost_matrix, tracks, detections, only_position=False,type_diff=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]*85         #cam9: 7, cam4: 85
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row]=gating_distance
    
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    if type_diff:
        for i in range(len(cost_matrix)):
            
            for j in range(len(cost_matrix[i])):
                if tracks[i].vehicle_type!='Undetermine' and tracks[i].vehicle_type!=detections[j].infer_type() :
                    if tracks[i].vehicle_type in ['motor','bicycle','person']:
                        cost_matrix[i][j]=np.inf
                    elif detections[j].infer_type() not in ['bus','truck']:
                        cost_matrix[i][j]=np.inf
            
           
    return cost_matrix

def gate_cost_matrix2(kf, cost_matrix, tracks, detections, only_position=False,type_diff=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]*85         #cam9: 7, cam4: 85
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row]=gating_distance
    
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    if type_diff:
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[i])):
                if tracks[i].vehicle_type!='Undetermine' and tracks[i].vehicle_type!=detections[j].infer_type() :
                    cost_matrix[i][j]=np.inf
            # if tracks[i].track_id==28:
            #     print(cost_matrix[i])
    return cost_matrix
def gate_cost_matrix3(kf, cost_matrix, tracks, detections, only_position=False,type_diff=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]*85         #cam9: 7, cam4: 85
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row]=gating_distance
    
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    if type_diff:
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[i])):
                if tracks[i].vehicle_type!='Undetermine' and tracks[i].vehicle_type!=detections[j].infer_type() and detections[j].infer_type() in ['bus','truck','car']:
                    cost_matrix[i][j]=np.inf
                       
    return cost_matrix

import os


    
    


def fuse_motion(opt,kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98,lost_map=None,occlusion_map=None,thres=None):
    if cost_matrix.size == 0:
        return cost_matrix
    
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    
  
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')#*(np.asarray(occlusion_map,dtype=np.int)*1000+1)

       
        
       
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        if lost_map==None or lost_map[row]==0:
            cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance 
        else:
            #Two effect, first is set higher score for lost track to be retracked, second is have more priority to current tracked track (first assosiate detect to all tracked
            #  track then conider remain box for lost track )
            cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance  -0.25+thres#-0.32+thres

    
    

    return cost_matrix
