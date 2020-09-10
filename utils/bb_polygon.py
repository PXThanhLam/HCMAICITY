#BTC DUA CODE SAI
# Idea:  
# 1) Draw a horizontal line to the right of each point and extend it to infinity

# 2) Count the number of times the line intersects with polygon edges.

# 3) A point is inside the polygon if either count of intersections is odd or
#    point lies on an edge of polygon.  If none of the conditions is true, then 
#    point lies outside.

# Given three colinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
import json
import numpy as np
from scipy.special import softmax
import numpy as np

def onSegment(p, q, r):
	if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
		return True 
	return False 


# To find orientation of ordered triplet (p, q, r). 
# The function returns following values 
# 0 --> p, q and r are colinear 
# 1 --> Clockwise 
# 2 --> Counterclockwise 
def orientation(p, q, r):
	val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
  
  	# colinear 
	if (val == 0):
		return 0  			

   	# clock or counterclock wise 
	if (val > 0):
		return 1
	else:
		return 2

def is_intersect(p1, q1, p2, q2):
	# Find the four orientations needed for general and special cases 
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1) 
	o4 = orientation(p2, q2, q1) 
  
	# General case 
	if (o1 != o2 and o3 != o4):
		return True 
  
	# Special Cases 
	# p1, q1 and p2 are colinear and p2 lies on segment p1q1 
	if (o1 == 0 and onSegment(p1, p2, q1)):
		return True
  
	# p1, q1 and p2 are colinear and q2 lies on segment p1q1 
	if (o2 == 0 and onSegment(p1, q2, q1)):
		return True
  
	# p2, q2 and p1 are colinear and p1 lies on segment p2q2 
	if (o3 == 0 and onSegment(p2, p1, q2)):
		return True 
  
	# p2, q2 and q1 are colinear and q1 lies on segment p2q2 
	if (o4 == 0 and onSegment(p2, q1, q2)):
		return True
  
	return False # Doesn't fall in any of the above cases

def is_point_in_polygon(polygon, point):
	# Create a point for line segment from p to infinite 
	point=[point[0]+1e-9,point[1]+1e-9]
	extreme = [point[0], 1e9]

	# Count intersections of the above line with sides of polygon 
	count = 0
	i = 0

	while True:
		j = (i+1) % len(polygon)

		# Check if the line segment from 'p' to 'extreme' intersects 
		# with the line segment from 'polygon[i]' to 'polygon[j]'
		if is_intersect(polygon[i], polygon[j], point, extreme):
			# If the point 'p' is colinear with line segment 'i-j', 
			# then check if it lies on segment. If it lies, return true, 
			# otherwise false 
			if orientation(polygon[i], point, polygon[j])==0:
				return onSegment(polygon[i], point, polygon[j])
			count = count + 1

		i = j
		if i==0:
			break
	return count % 2 == 1


def is_bounding_box_intersect(bounding_box, polygon):
	for i in range(len(bounding_box)):
		if is_point_in_polygon(polygon, bounding_box[i]):
			return True
	return False

def check_bbox_intersect_or_outside_polygon(polygon, bbox):
    x1, y1, x2, y2 = bbox
    bb = [(x1,y1), (x2, y1), (x2,y2), (x1,y2)]
    for i in range(len(bb)):
	    if not is_point_in_polygon(polygon, bb[i]):
		    return True
    return False
 
def check_bbox_outside_polygon(polygon, bbox):
	x1, y1, x2, y2 = bbox
	bb = [(x1,y1), (x2, y1), (x2,y2), (x1,y2), ((x1+x2)/2,(y1+y2)/2) , ((x1+x2)/2,(0.7*y1+0.3*y2)), ((x1+x2)/2,(0.3*y1+0.7*y2)) ]
	for i in range(len(bb)):
		if is_point_in_polygon(polygon, bb[i]):
			return False
	return True
def check_bbox_inside_polygon(polygon, bbox):
	x1, y1, x2, y2 = bbox
	bb = [(x1,y1), (x2, y1), (x2,y2), (x1,y2)]
	for i in range(len(bb)):
		if not is_point_in_polygon(polygon, bb[i]):
			return False
	return True
def cosin_similarity(a2d, b2d):
	a=np.array((a2d[1][0] - a2d[0][0], a2d[1][1]- a2d[0][1]))
	b=np.array((b2d[1][0] - b2d[0][0], b2d[1][1] - b2d[0][1]))
	return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def load_zone_anno(json_filename):
	with open(json_filename) as jsonfile:
		dd = json.load(jsonfile)
		polygon = [(int(x), int(y)) for x, y in dd['shapes'][0]['points']]
		paths = {}
		for it in dd['shapes'][1:]:
			kk = str(int(it['label'][-2:]))
			paths[kk] = [(int(x), int(y)) for x, y in it['points']]
	return polygon, paths

def counting_moi(paths,vector_list):
	moi_detection_list = []
	for vector in vector_list:
		max_cosin = -2
		movement_id = ''
		last_frame = 0
		for movement_label, movement_vector in paths.items():
			cosin = cosin_similarity(movement_vector, vector)
			if cosin > max_cosin:
				max_cosin = cosin
				movement_id = movement_label
		moi_detection_list.append(movement_id)
	return moi_detection_list
def point_to_line_distance(point,line):
	p1,p2=line
	p3=point
	return abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
def tlbrs_to_mean_area(tlbrs):
	areas=[np.abs((x[2]-x[0])*(x[3]-x[1])) for x in tlbrs]
	return np.mean(areas)
	

			
def find_best_fit_line(points, paths):
	movement_id = ''
	movement_voting_list = [0]*len(paths.keys())
	for i in range(len(points)-1):
		direction_prob = []
		for movement_label, movement_vector in paths.items():
		
			track_vector = (points[i], points[i+1])
			cosin = cosin_similarity(track_vector, movement_vector)
			direction_prob.append(cosin)
		movement_temp_id = np.argmax(softmax(np.array(direction_prob)))
		movement_voting_list[movement_temp_id] +=1
	return np.argmax(np.array(movement_voting_list))
def lineFromPoints(P,Q): 
	a = Q[1] - P[1] 
	b = P[0] - Q[0]  
	c =-(a*(P[0]) + b*(P[1]))
	return a,b,c

#bot,on, up:0,1,2
def point_line_relative(point,line):
	a,b,c=lineFromPoints(line[0],line[1])
	# print(a,b,c)
	x,y=point
	direction=a*x+b*y+c
	relative_pos = 0 if direction<0 else 1 if direction==0 else 2
	# print('=-----------')
	# print(point)
	# print(line)
	# print(relative_pos)
	# print('=----------')
	return relative_pos
#t,l,b,r bbox
def box_line_relative(box,line): 
	t,l,b,r=box
	p1,p2,p3,p4=(t,l),(b,l),(t,r),(b,r)
	rela1,rela2,rela3,rela4=point_line_relative(p1,line),point_line_relative(p2,line),point_line_relative(p3,line),point_line_relative(p4,line)
	bottom_num=0
	up_num=0
	for rela in[rela1,rela2,rela3,rela4]:
		if rela==0:
			bottom_num+=1
		if rela==2:
			up_num+=1
	pos= 'up' if bottom_num==0 else 'bottom' if up_num==0 else 'cross'
	return pos

if __name__=='__main__':
	# points=[     704.68    ,  585.05      ,788.67      ,703.99]
	# polygon=[
    #             [
    #                 129.25806451612902,
    #                 637.8709677419355
    #             ],
    #             [
    #                 97.806451612903224,
    #                 190.09677419354838
    #             ],
    #             [
    #                 501.0,
    #                 152.5483870967742
    #             ],
	# 	[
    #                 679.0,
    #                 238.5483870967742
    #             ]
    #              ,
    #             [
    #                920,
    #                251
    #             ]
	# 	,
    #             [
    #                 1279.2903225806451,
    #                 442.2258064516129
    #             ]
    #         ]
	# print(check_bbox_intersect_or_outside_polygon(polygon,points))
	
	point= [    
               [
                   338.5, 294.5
                ],
				[261.5, 260.5]
			]
	
	line3=[
                [     455.0,
                    238.47619047619048
                ],
                [
                    193.09523809523807,
                    325.1428571428571
                ]
            ]
            
	line5= [
                [
                    936.9047619047618,
                    675.6190476190476
                ],
                [
                    171.19047619047618,
                    349.90476190476187
                ]
            ]
	line8=[(55, 510), (1178, 388)]
	line10=[(726,256), (1139, 388)] #[(533, 238), (1156, 381)]
	line1=[(491, 244), (355, 678)]
	line12=[(1165, 247), (375, 693)] #[(1165, 397), (375, 693)]
	point2=[[
                    0.7142857142857144,
                    656.5714285714286
                ],[
                    1278.8095238095239,
                    660.3809523809523
                ]]
    
	
	point3=[[892,415],[692,655]]
			 
	print(cosin_similarity(point3,line1))
	print(cosin_similarity(point3,line12))
	print(cosin_similarity(line1,line12))

	
