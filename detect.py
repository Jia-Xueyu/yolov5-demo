import torch
import torch.nn as nn
from utils.datasets import LoadImages
import numpy as np
import argparse
import os
from model.yolo import yolo
from utils.loss import ComputeLoss
from utils.general import nms,xywh2xyxy,draw_img

gpus=[0]
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIABLE_DEVICES']=','.join(map(str,gpus))

def write_result(pred,path,name):
    img_path=os.path.join(path,name)
    cls=np.argmax(pred[:,5:],1)
    with open(img_path,'w') as f:
        for i in range(pred.shape[0]):
            f.write(str(cls[i])+' '.join(map(str,pred[i,:5]))+'\n')

    print('Write Done!')

def remove_padding(letterbox_pred,padding_info,ori_img):
    '''
    将变化后的图片转化到原来的尺寸
    '''


def train(opt):
    datasets=LoadImages(opt.path,opt.img_sz)
    model=torch.load('./runs/model.pt')
    model.eval()
    for i,img_name,letterbox_img,padding in datasets:
        img=np.expand_dims(letterbox_img,0)
        img=torch.tensor(img).float().cuda()
        pred=model(img)
        pred=pred[0,:,:].detach().cpu().numpy()
        pred_xyxy=xywh2xyxy(pred)
        nms_pred=nms(pred_xyxy,opt.iou_thres)
        conf_pred=nms_pred[np.where(nms_pred[:,4]>opt.conf_thres)]
        # write_result(nms_pred,opt.save_path,img_name.split('.')[0]+'.txt')
        draw_img(letterbox_img.transpose(1,2,0),conf_pred,1,True)

    print('Done!')





if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='./data/',help='detect data path')
    parser.add_argument('--save_path',type=str,default='./runs/test/',help='save path')
    parser.add_argument('--iou_thres',type=float,default=0.001,help='nms iou threshold')
    parser.add_argument('--conf_thres',type=float,default=0.999999,help='confidence threshold')
    parser.add_argument('--img_sz', type=int, default=640, help='image size')
    opt=parser.parse_args()
    train(opt)