import torch
import torch.nn as nn
from utils.datasets import create_dataloader
import numpy as np
import argparse
import os
from model.yolo import yolo
from utils.loss import ComputeLoss
from torchsummary import summary
from tqdm import tqdm
from torch.autograd import Variable
from utils.general import draw_img

gpus=[0]
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIABLE_DEVICES']=','.join(map(str,gpus))


def train(opt,hyp):
    dataloader,data_len=create_dataloader(opt.path,opt.size,opt.batch)
    model=yolo(opt.cfg,hyp).cuda()
    loss_criterion=ComputeLoss(model)
    optimizer=torch.optim.Adam(lr=opt.lr,params=model.parameters())
    # summary(model,(3,640,640))
    for i in range(opt.epochs):
        train_loss=0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for j,(imgs,targets) in pbar:
            '''
            可视化转化后的图片信息
            '''
            # temp=np.ascontiguousarray(imgs[0].numpy().transpose(1,2,0))
            # temp_t=targets[np.where(targets[:,0]==0)].numpy()
            # draw_img(temp,temp_t,640)


            imgs=Variable(imgs.cuda().type(torch.cuda.FloatTensor))
            targets=Variable(targets.cuda().type(torch.cuda.FloatTensor))
            pred=model(imgs)
            loss=loss_criterion(pred,targets)
            train_loss+=loss.detach().cpu().numpy()[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j==len(pbar)-1:
                l=data_len
            else:
                l=(j+1)*imgs.shape[0]
            pbar.set_description('epoch: {}, train loss: {:8.3f}'.format(i,train_loss/l))
    torch.save(model,'./runs/model.pt')




if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='./data/',help='train data and labels path')
    parser.add_argument('--batch',type=int,default=3)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--lr',type=float,default=1e-3,help='learning rate')
    parser.add_argument('--size',type=int,default=640,help='image size')
    parser.add_argument('--cfg',type=str,default='./model/model.yaml',help='model config')
    opt=parser.parse_args()

    hyp={
        'fl_gamma':1.5,
        'anchor_t':4,
        'box':1.0,
        'cls':1.0,
        'obj':1.0,
    }
    train(opt,hyp)