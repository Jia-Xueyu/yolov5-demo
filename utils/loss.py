import torch
import torch.nn as nn
from utils.general import bbox_iou


def smooth_BCE(eps=0.1):
    return 1.0-0.5*eps,eps*0.5
class FocalLoss(nn.Module):
    def __init__(self,loss_fnc=nn.BCEWithLogitsLoss,alpha=0.25,gamma=2):
        super(FocalLoss,self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.loss_func=loss_fnc
        self.reduction=self.loss_func.reduction
        self.loss_func.reduction='none'

    def forward(self,pred,true):
        loss=self.loss_func(pred,true)
        alpha_factor=true*self.alpha+(1-true)*(1-self.alpha)
        pred_prob=torch.sigmoid(pred)
        modulating_factor=torch.abs(1-pred_prob)**self.gamma
        loss*=alpha_factor*modulating_factor
        if self.reduction =='mean':
            return loss.mean()
        elif self.reduction =='sum':
            return loss.sum()
        else:
            return loss

class ComputeLoss:
    def __init__(self,model):
        super(ComputeLoss,self).__init__()
        BCE_obj=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0).cuda())
        BCE_cls=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0).cuda())
        self.hyp=model.hyp
        self.cp,self.cn=smooth_BCE(0.0)
        self.BCE_obj=FocalLoss(BCE_obj,gamma=self.hyp['fl_gamma'])
        self.BCE_cls=FocalLoss(BCE_cls,gamma=self.hyp['fl_gamma'])
        det=model.model[-1]
        for i in 'nl','na','nc','anchors':
            setattr(self,i,getattr(det,i))

    def __call__(self,pred,target):#pred bs,3,80,80,85,target: image,cls,xywh
        cls_loss,box_loss,obj_loss=torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda()
        tcls,tbox,indices,anch=self.build_target(pred,target)

        for i,pi in enumerate(pred):
            b,a,gj,gi=indices[i]
            tobj=torch.zeros_like(pi[...,0])
            n=b.shape[0]
            if n:
                ps=pi[b,a,gj,gi]
                pxy=ps[:,0:2].sigmoid()*2-0.5
                pwh=(ps[:,2:4].sigmoid()*2)**2*anch[i]
                pbox=torch.cat((pxy,pwh),1)
                iou=bbox_iou(pbox,tbox[i],GIOU=True)
                box_loss+=(1.0-iou).mean()

                tobj[b,a,gj,gi]=iou.detach().clamp(0).type(tobj.dtype)

                if self.nc>1:
                    t=torch.full_like(ps[:,5:],self.cn)
                    t[range(n),tcls[i]]=self.cp
                    cls_loss+=self.BCE_cls(ps[:,5:],t)

            obj_loss+=self.BCE_obj(pi[...,4],tobj)

        box_loss*=self.hyp['box']
        obj_loss*=self.hyp['obj']
        cls_loss*=self.hyp['cls']

        loss=cls_loss+obj_loss+box_loss
        return loss


    def build_target(self,pred,target):
        bs, na=target.shape[0],self.na
        tcls,tbox,indices,anch=[],[],[],[]
        grid_gain=torch.ones(7).cuda()
        ai=torch.arange(na).float().view(na,1).repeat(1,bs).cuda()
        target=torch.cat((target.repeat(na,1,1),ai[:,:,None]),2)

        g=0.5
        off=torch.tensor([[0,0],[1,0],[0,1],[-1,0],[0,-1]]).float().cuda()*g

        for i in range(self.nl):
            anchors=self.anchors[i]
            grid_gain[2:6]=torch.tensor(pred[i].shape)[[3,2,3,2]].cuda()

            t=target*grid_gain

            if bs:
                r=t[...,4:6]/anchors[:,None]
                j=torch.max(r,1/r).max(2)[0]<self.hyp['anchor_t']
                t=t[j]

                gxy=t[:,2:4]
                gxi=grid_gain[[2,3]]-gxy
                j,k=((gxy%1<g)&(gxy>1)).T
                l,m=((gxi%1<g)&(gxi>1)).T
                j=torch.stack((torch.zeros_like(j),j,k,l,m))
                t=t.repeat((5,1,1))[j]
                offset=(torch.zeros_like(gxy)[None]+off[:,None])[j]
            else:
                offset=0
                t=target[0]

            b,c=t[:,0:2].long().T
            gxy=t[:,2:4]
            gwh=t[:,4:6]
            gij=(gxy-offset).long()
            i,j=gij.T

            a=t[:,6].long()
            indices.append((b,a,j.clamp_(0,grid_gain[3]-1),i.clamp_(0,grid_gain[2]-1)))
            anch.append(anchors[a])
            tbox.append(torch.cat((gxy-gij.float(),gwh),1))
            tcls.append(c)

        return tcls,tbox,indices,anch

