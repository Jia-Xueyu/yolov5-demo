import torch
import numpy as np
import cv2

def bbox_iou(box1,box2,GIOU=True):  #box1 (4,bs), box2 (bs,4), xywh
    box2=box2.T
    box1=box1.T
    b1_x1,b1_y1=box1[0]-box1[2]/2,box1[1]-box1[3]/2
    b1_x2,b1_y2=box1[0]+box1[2]/2,box1[1]+box1[3]/2
    b2_x1,b2_y1=box2[0]-box2[2]/2,box2[1]-box2[3]/2
    b2_x2,b2_y2=box2[0]+box2[2]/2,box2[1]+box2[3]/2

    inter=(torch.min(b1_x2,b2_x2)-torch.max(b1_x1,b2_x1)).clamp(0)*(torch.min(b1_y2,b2_y2)-torch.max(b1_y1,b2_y1)).clamp(0)

    union=(b1_x2-b1_x1)*(b1_y2-b1_y1)+(b2_x2-b2_x1)*(b2_y2-b2_y1)-inter

    convex_w=torch.max(b1_x2,b2_x2)-torch.min(b1_x1,b2_x1)
    convex_h=torch.max(b1_y2,b2_y2)-torch.min(b1_y1,b2_y1)

    iou=inter/union
    convex=convex_h*convex_w
    Giou=iou-(convex-union)/convex
    return Giou if GIOU else iou


def nms(detection,iou_threshold=0.45):
    if detection.shape==0:
        return detection
    left_x=detection[:,0]
    left_y=detection[:,1]
    right_x=detection[:,2]
    right_y=detection[:,3]
    area=(right_x-left_x+1)*(right_y-left_y+1)
    order=np.argsort(detection[:,4])
    keep=[]
    while order.size>0:
        i=order[-1]
        others=order[:-1]
        keep.append(np.expand_dims(detection[i],0))
        w=np.minimum(right_x[i].repeat(len(right_x[others])),right_x[others])-np.maximum(left_x[i].repeat(len(left_x[others])),left_x[others])
        w=np.maximum(w,0)
        h=np.minimum(right_y[i].repeat(len(right_y[others])),right_y[others])-np.maximum(left_y[i].repeat(len(left_y[others])),left_y[others])
        h=np.maximum(h,0)
        inter=w*h
        iou=inter/(area[i]+area[others]-inter)
        less_thres_index=np.where(iou<iou_threshold)
        order=order[less_thres_index]
    return np.concatenate(keep,0)

def xywh2xyxy(det):
    left_x=det[:,0]-det[:,2]/2
    left_y=det[:,1]-det[:,3]/2
    right_x=det[:,0]+det[:,2]/2
    right_y=det[:,1]+det[:,3]/2
    det[:,0],det[:,1],det[:,2],det[:,3]=left_x,left_y,right_x,right_y
    return det

def draw_img(img,targets,img_sz,test=False):
    if isinstance(img_sz,int):
        img_sz=(img_sz,img_sz)
    if not test:
        left_x=(targets[:,2]-targets[:,4]/2)*img_sz[1]
        left_y=(targets[:,3]-targets[:,5]/2)*img_sz[0]
        right_x=(targets[:,2]+targets[:,4]/2)*img_sz[1]
        right_y=(targets[:,3]+targets[:,5]/2)*img_sz[0]
    else:
        left_x,left_y,right_x,right_y=targets[:,0],targets[:,1],targets[:,2],targets[:,3]
        cls = np.argmax(targets[:, 5:], 1)
    left_x,left_y,right_x,right_y=left_x.astype(int),left_y.astype(int),right_x.astype(int),right_y.astype(int)

    left_x=np.minimum(np.maximum(left_x, 0), img.shape[1])
    left_y = np.minimum(np.maximum(left_y, 0), img.shape[0])
    right_x = np.minimum(np.maximum(right_x, 0), img.shape[1])
    right_y = np.minimum(np.maximum(right_y, 0), img.shape[0])

    for i in range(left_x.shape[0]):
        text=str(int(targets[i,1])) if not test else str(cls[i])+' conf: {:.2f}'.format(targets[i,4])

        img=cv2.rectangle(img,(left_x[i],left_y[i]),(right_x[i],right_y[i]),color=(0,0,255),thickness=2)
        img=cv2.putText(img,text,(left_x[i],left_y[i]),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
    cv2.imshow('pic',img)
    key=cv2.waitKey(0)
    cv2.destroyAllWindows()
