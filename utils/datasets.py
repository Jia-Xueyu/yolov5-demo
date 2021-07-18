import torch
from torch.utils.data import Dataset,DataLoader
import os
import cv2
import numpy as np


def create_dataloader(path,imgsz,batch):
    datasets=LoadImageAndLabel(path,imgsz,batch)
    batch=min(batch,len(datasets))
    dataloader=DataLoader(datasets,batch_size=batch,shuffle=True,collate_fn=LoadImageAndLabel.collate_fn)
    return dataloader,len(datasets)

def letterbox(img,new_sz=(640,640),color=(141,141,141)):
    shape=img.shape[:2]
    if isinstance(new_sz,int):
        new_sz=(new_sz,new_sz)
    r=min(new_sz[0]/shape[0],new_sz[1]/shape[1])
    r_h,r_w=shape[0]*r,shape[1]*r
    h_pad,w_pad=new_sz[0]-r_h,new_sz[0]-r_w
    top_pad=int(h_pad//2)
    bottom_pad=int(new_sz[0]-shape[0]-top_pad)
    left_pad=int(w_pad//2)
    right_pad=int(new_sz[1]-shape[1]-left_pad)
    img=cv2.copyMakeBorder(img,top_pad,bottom_pad,left_pad,right_pad,cv2.BORDER_CONSTANT,value=color)
    return img,(top_pad,bottom_pad,left_pad,right_pad)

def revise_label(padding,label,img_shape,index):
    labels=[]
    new_h=img_shape[0]+padding[0]+padding[1]
    new_w=img_shape[1]+padding[2]+padding[3]
    for i in label:
        x=(i[1]*img_shape[1]+padding[2])/new_w
        y=(i[2]*img_shape[0]+padding[0])/new_h
        w=i[3]*img_shape[1]/new_w
        h=i[4]*img_shape[0]/new_h
        labels.append([index,i[0],x,y,w,h])
    labels=np.array(labels)
    return labels


class LoadImageAndLabel(Dataset):
    def __init__(self,path,imgsz=640,batch=16,stride=1):#path下同时有train和label文件夹，图片和label同名
        self.img_path=os.path.join(path,'images')
        self.label_path=os.path.join(path,'labels')
        self.imgsz=imgsz
        self.batch=batch
        self.stride=stride
        self.imgs=[]
        for i in os.listdir(self.img_path):
            self.imgs.append(i)
        self.imgs.sort(key=lambda x:os.path.splitext(x)[0])
        self.labels=[]
        for i in os.listdir(self.label_path):
            self.labels.append(i)
        self.labels.sort(key=lambda x:os.path.splitext(x)[0])
        assert [i.split('.')[0] for i in self.imgs]==[j.split('.')[0] for j in self.labels], 'imgs do not match labels'

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        label_files=self.labels[index]
        temp=os.path.join(self.label_path,label_files)
        with open(temp,'r') as f:
            data=f.readlines()
            data=[d.strip('\n').split(' ') for d in data]
            data=[[float(d[j]) if j>0 else int(d[j]) for j in range(len(d))] for d in data ]
            labels=data


        img_files=self.imgs[index]

        img_data=cv2.imread(os.path.join(self.img_path,img_files))
        r=min(self.imgsz/img_data.shape[0],self.imgsz/img_data.shape[1])

        img=cv2.resize(img_data,(int(r*img_data.shape[1]),int(r*img_data.shape[0])),interpolation=cv2.INTER_LINEAR)
        img_shape = img.shape[:2]
        img,padding=letterbox(img,new_sz=self.imgsz)


        #revise the label
        new_label=revise_label(padding,labels,img_shape,index)

        img=img.transpose(2,0,1)
        return torch.from_numpy(img),torch.from_numpy(new_label)

    @staticmethod
    def collate_fn(batch):
        img,label=zip(*batch)
        for i,l in enumerate(label):
            if len(l)>0:
                l[:,0]=i
        return torch.stack(img,0),torch.cat(label,0)


class LoadImages:
    def __init__(self,path,img_sz):
        self.img_path = os.path.join(path, 'test')
        self.imgs=[]
        self.img_sz=img_sz
        for i in os.listdir(self.img_path):
            self.imgs.append(i)

    def __iter__(self):
        self.count=0
        return self

    def __next__(self):
        if self.count==len(self.imgs):
            raise StopIteration
        img=os.path.join(self.img_path,self.imgs[self.count])
        img_data=cv2.imread(img)

        self.count+=1

        #resize and padding
        r=min(self.img_sz/img_data.shape[0],self.img_sz/img_data.shape[1])
        img_data=cv2.resize(img_data,(int(r*img_data.shape[1]),int(r*img_data.shape[0])),interpolation=cv2.INTER_LINEAR)
        letterbox_img,padding_info=letterbox(img_data)

        img_data=img_data.transpose(2,0,1)
        letterbox_img=letterbox_img.transpose(2,0,1)

        return img_data,self.imgs[self.count],letterbox_img,padding_info





