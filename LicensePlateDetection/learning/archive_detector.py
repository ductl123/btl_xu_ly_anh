"""Small, single-class plate detector for local CPU training and inference."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SIZE=320
STRIDE=8


def conv(a,b,stride=1,k=3):
    return nn.Sequential(nn.Conv2d(a,b,k,stride,k//2,bias=False),nn.BatchNorm2d(b),nn.SiLU())


class PlateDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(conv(3,16,2),conv(16,24,2),conv(24,32,2))
        self.mid=nn.Sequential(conv(32,48,2),conv(48,48))
        self.deep=nn.Sequential(conv(48,64,2),conv(64,64))
        self.lateral=conv(64,48,k=1)
        self.fuse=nn.Sequential(conv(80,48),conv(48,32))
        self.head=nn.Conv2d(32,5,1)
        with torch.no_grad():
            self.head.bias[0]=-3.5;self.head.bias[1:]=0
    def forward(self,x):
        a=self.stem(x);b=self.mid(a);c=self.deep(b)
        b=b+F.interpolate(self.lateral(c),size=b.shape[-2:],mode='nearest')
        return self.head(self.fuse(torch.cat((a,F.interpolate(b,size=a.shape[-2:],mode='nearest')),1)))


def read_rgb(path):
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:raise ValueError('Cannot read '+str(path))
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def letterbox(image,size=SIZE):
    h,w=image.shape[:2];scale=min(size/w,size/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    dx,dy=(size-nw)//2,(size-nh)//2
    canvas=np.full((size,size,3),114,np.uint8)
    canvas[dy:dy+nh,dx:dx+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    return canvas,(nw/w,nh/h,dx,dy)


def yolo_to_xyxy(boxes,w,h):
    a=np.asarray(boxes,dtype=np.float32).reshape(-1,5)[:,1:]
    a=a*np.array([w,h,w,h],np.float32)
    return np.concatenate([a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2],1).clip([0,0,0,0],[w,h,w,h]).astype(np.float32)


def targets(boxes,size=SIZE):
    g=size//STRIDE;result=np.zeros((5,g,g),np.float32)
    yy,xx=np.mgrid[:g,:g]
    # Largest first so a smaller plate wins if two centers share one cell.
    for x0,y0,x1,y1 in sorted(boxes,key=lambda b:-(b[2]-b[0])*(b[3]-b[1])):
        cx,cy=(x0+x1)/(2*STRIDE),(y0+y1)/(2*STRIDE)
        w,h=max(.15,(x1-x0)/STRIDE),max(.15,(y1-y0)/STRIDE)
        ix,iy=min(g-1,int(cx)),min(g-1,int(cy))
        sigma=max(.55,min(2.5,min(w,h)*.24))
        result[0]=np.maximum(result[0],np.exp(-((xx-ix)**2+(yy-iy)**2)/(2*sigma*sigma)))
        result[1:,iy,ix]=[cx-ix,cy-iy,np.log(w),np.log(h)]
    return result


def loss_fn(pred,target):
    heat=target[:,0];positive=heat==1
    p=pred[:,0].sigmoid().clamp(1e-5,1-1e-5)
    focal=-(torch.log(p)*(1-p).square()*positive+torch.log1p(-p)*p.square()*(1-heat).pow(4)*~positive).sum()/positive.sum().clamp_min(1)
    a=pred[:,1:].permute(0,2,3,1)[positive];b=target[:,1:].permute(0,2,3,1)[positive]
    if len(a)==0:return focal
    offset=a[:,:2].sigmoid();wh=a[:,2:].clamp(-3,5).exp();truth=b[:,2:].exp()
    tl=torch.maximum(offset-wh/2,b[:,:2]-truth/2);br=torch.minimum(offset+wh/2,b[:,:2]+truth/2)
    overlap=(br-tl).clamp_min(0).prod(1)
    union=wh.prod(1)+truth.prod(1)-overlap
    iou=(overlap/union.clamp_min(1e-6)).mean()
    return focal+2*F.smooth_l1_loss(offset,b[:,:2])+F.smooth_l1_loss(a[:,2:],b[:,2:])+2*(1-iou)


def decode(output,score=.05,top_k=100):
    heat=output[:,0:1].sigmoid()
    heat=heat*(heat==F.max_pool2d(heat,3,1,1))
    k=min(top_k,heat.shape[-2]*heat.shape[-1]);scores,index=heat.flatten(1).topk(k)
    g=heat.shape[-1];offset=output[:,1:3].flatten(2).gather(2,index[:,None].expand(-1,2,-1)).sigmoid()
    size=output[:,3:5].flatten(2).gather(2,index[:,None].expand(-1,2,-1)).clamp(-3,5).exp()
    centers=(torch.stack((index%g,index//g),1)+offset)*STRIDE
    size=size*STRIDE
    boxes=torch.cat((centers-size/2,centers+size/2),1).transpose(1,2).detach().cpu().numpy()
    scores=scores.detach().cpu().numpy();result=[]
    for b,s in zip(boxes,scores):
        valid=s>=score;b=b[valid];s=s[valid]
        xywh=np.concatenate([b[:,:2],b[:,2:]-b[:,:2]],1)
        keep=cv2.dnn.NMSBoxes(xywh.tolist(),s.tolist(),float(score),.45)
        indices=np.asarray(keep,dtype=int).reshape(-1)
        result.append(np.column_stack((b[indices],s[indices])).astype(np.float32))
    return result


def map_to_original(pred,geometry,w,h):
    sx,sy,x,y=geometry;result=pred.copy()
    result[:,[0,2]]=(result[:,[0,2]]-x)/sx;result[:,[1,3]]=(result[:,[1,3]]-y)/sy
    result[:,:4]=result[:,:4].clip([0,0,0,0],[w,h,w,h])
    return result[(result[:,2]>result[:,0])&(result[:,3]>result[:,1])]


def load_detector(path):
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
    if checkpoint.get('architecture')!='archive_plate_detector_v1':raise ValueError('Wrong detector checkpoint')
    model=PlateDetector();model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint


@torch.inference_mode()
def predict(model,image,score=.20):
    array,geometry=letterbox(image)
    batch=torch.from_numpy(array).permute(2,0,1)[None].float()/255.
    return map_to_original(decode(model(batch),score=score)[0],geometry,image.shape[1],image.shape[0])
