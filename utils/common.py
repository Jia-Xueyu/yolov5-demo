import torch
import torch.nn as nn



def autopad(k,q=None):
    if q is None:
        q=k//2 if isinstance(k,int) else [i//2 for i in k]
    return q


class Module_ch(nn.Module):
    def __init__(self,c2=3):
        super(Module_ch,self).__init__()
        self.c2=c2
    def ch_num(self):
        return self.c2

class Conv(Module_ch):
    def __init__(self,c1,c2,k=1,s=1,q=None):
        super(Conv,self).__init__(c2)
        self.cv=nn.Conv2d(c1,c2,k,s,padding=autopad(k,q))
        self.bn=nn.BatchNorm2d(c2)
        self.act=nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.bn(self.cv(x)))

class Res(Module_ch):
    def __init__(self,c1,c2,skip_connect=True,hidden_factor=0.5):
        super(Res,self).__init__(c2)
        c_hidden=int(hidden_factor*c2)
        self.conv1=Conv(c1,c_hidden,1,1)
        self.conv2=Conv(c_hidden,c2,3,1)
        self.add=skip_connect and c1==c2

    def forward(self,x):
        return x+self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class Concat(nn.Module):
    def __init__(self,concat_dimension=1):
        super(Concat,self).__init__()
        self.d=concat_dimension

    def forward(self,x):
        return torch.cat(x,self.d)

class BottleneckCSP(Module_ch):
    def __init__(self,c1,c2,block_num=1,hidden_factor=0.5):
        super(BottleneckCSP,self).__init__(c2)
        c_hidden=int(hidden_factor*c2)
        self.conv1=Conv(c1,c_hidden,1,1)
        self.conv2=nn.Conv2d(c_hidden,c_hidden,1,1)
        self.conv3=nn.Conv2d(c1,c_hidden,1,1)
        self.res=nn.Sequential(*[Res(c_hidden,c_hidden) for _ in range(block_num)])
        self.concat=Concat()
        self.bn=nn.BatchNorm2d(c_hidden*2)
        self.act=nn.LeakyReLU(0.1)
        self.conv4=Conv(c_hidden*2,c2,1,1)

    def forward(self,x):
        y1=self.conv2(self.res(self.conv1(x)))
        y2=self.conv3(x)
        y=self.concat([y1,y2])
        return self.conv4(self.act(self.bn(y)))

class SPP(Module_ch):
    def __init__(self,c1,c2,k=(5,7,9)):
        super(SPP,self).__init__(c2)
        c_=c1//2
        self.conv1=Conv(c1,c_,1,1)
        self.conv2=Conv(c_*(len(k)+1),c2,1,1)
        self.m=nn.ModuleList([nn.MaxPool2d(kernel_size=i,stride=1,padding=i//2) for i in k])

    def forward(self,x):
        y1=self.conv1(x)
        y2=torch.cat([y1]+[m(y1) for m in self.m],1)
        return self.conv2(y2)

class Focus(Module_ch):
    def __init__(self,c1,c2):
        super(Focus,self).__init__(c2)
        self.conv=Conv(c1*4,c2,1,1)

    def forward(self,x):
        return self.conv(torch.cat([x[...,::2,::2],x[...,1::2,::2],x[...,::2,1::2],x[...,1::2,1::2]],1))

class Detect(nn.Module):
    stride=None
    def __init__(self,anchors=(),nc=80,ch=()):
        super(Detect,self).__init__()
        self.nl=len(anchors)
        self.nc=nc
        self.no=nc+5
        self.na=len(anchors[0])//2
        self.grid=[torch.zeros(1)]*self.nl
        a=torch.tensor(anchors).float().cuda().view(self.nl,-1,2)
        self.register_buffer('anchors',a)
        self.register_buffer('anchor_grid',a.clone().view(self.nl,1,-1,1,1,2))
        self.m=nn.ModuleList([nn.Conv2d(x,self.no*self.nl,1,1) for x in ch])

    def forward(self,x):

        z=[]
        for i in range(self.nl):
            x[i]=self.m[i](x[i])
            bs,_,ny,nx=x[i].shape # batch,255,20,20 -> batch,3,20,20,85
            x[i]=x[i].view(bs,self.na,self.no,ny,nx).permute(0,1,3,4,2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4]!=x[i].shape[2:4]:
                    self.grid[i]=self._make_grid(nx,ny).cuda()
                y=x[i].sigmoid()
                y[...,0:2]=(y[...,0:2]*2-0.5+self.grid[i])*self.stride[i]
                y[...,2:4]=(y[...,2:4]*2)**2*self.anchor_grid[i]
                z.append(y.view(bs,-1,self.no))
        return x if self.training else torch.cat(z,1)

    @staticmethod
    def _make_grid(nx=20,ny=20):
        yv,xv=torch.meshgrid(torch.arange(ny),torch.arange(nx))
        return torch.stack((xv,yv),2).view((1,1,ny,nx,2)).float()