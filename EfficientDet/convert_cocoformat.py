import os
import cv2


def convert_coco(img_paths):
    for img_path in img_paths:
        h,w,_=cv2.imread(img_path).shape
        if os.path.exists(img_path[:-4]+'.txt'):
            img=cv2.imread(img_path)
            f = open('phone_data/coco_reformat/'+img_path[:-4].replace('/','_')+'.txt', "w")
            with open(img_path[:-4]+'.txt') as bbox_file:
                bboxs_coco=bbox_file.readlines()
                for bbox_coco in bboxs_coco:
                    bbox_coco=bbox_coco[:-1]
                    try:
                        _,pcx,pcy,pw,ph=bbox_coco.split(' ')
                        pcx,pcy,pw,ph=float(pcx),float(pcy),float(pw),float(ph)
                        pcx,pcy,pw,ph=int(pcx*w),int(pcy*h),int(w*pw),int(h*ph)
                        x1,y1,x2,y2=pcx-pw//2,pcy-ph//2,pcx+pw//2,pcy+ph//2
                        f.write(str(67)+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+'\n')
                    except:
                        pass           

                        
                        
                    
                    
            

        