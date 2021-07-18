from utils.common import *
from ruamel import yaml
import numpy as np

class yolo(nn.Module):
    def __init__(self,cfg,hyp):
        super(yolo,self).__init__()
        self.cfg=cfg
        self.hyp=hyp
        self.model,self.layer_from=parse_model(self.cfg)
        self.save_layer=np.unique([i for i in self.layer_from if isinstance(i,int)] + [j for i in self.layer_from if isinstance(i,list) for j in i])
        m=self.model[-1]
        s=640
        ch=3
        m.stride=torch.tensor([s/x.shape[-2] for x in self.forward(torch.zeros(1,ch,s,s))]).cuda()
        m.anchors/=m.stride.view(-1,1,1)

    def forward(self,x):
        save_dict={}
        for i in range(len(self.model)):
            m=self.model[i]
            if self.layer_from[i]==-1:
                x = m(x)
                if i in self.save_layer:
                    save_dict[i]=x
            else:
                concat_y=[]
                for j in self.layer_from[i]:
                    if j==-1:
                        concat_y.append(x)
                    else:
                        concat_y.append(save_dict[j])
                x=m(concat_y)
        return x




def parse_model(cfg):
    with open(cfg,'r') as f:
        model_dict=yaml.safe_load(f)

    model=[]
    layer_from=[]
    ch_num=[]
    for i,(f,n,m,args) in enumerate(model_dict['backbone']+model_dict['head']):
        layer_from.append(f)
        m=eval(m)
        if len(model)>=1:
            count=-1
            while type(model[count]) in [nn.Upsample,Concat]:
                count-=1

            c1=model[count].ch_num() if type(model[-1]) not in [Concat] else model[count].ch_num()*len(layer_from[-2])
            ch_num.append(c1)
        if m==Focus:
            model.append(m(args[1],args[0]))
        elif m==Conv:
            model.append(m(c1,*args))
        elif m==BottleneckCSP:
            model.append(m(c1,args[0],n))
        elif m==SPP:
            model.append(m(c1,args[0],args[1]))
        elif m==Detect:
            model.append(m(args[1],args[0],[ch_num[c] for c in f]))
        else:
            model.append(m(*args))

    yolo_model=nn.Sequential(*[x for x in model])
    return yolo_model,layer_from#,[ch_num[i] for i in layer_from[-1]]
