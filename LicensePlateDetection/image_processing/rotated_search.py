import cv2
import numpy as np
from utils.image_utils import order_quad_points
from .plate_verification import ink_layout

def rotated_proposals(image,max_dim=1400):
    H,W=image.shape[:2];scale=min(1,max_dim/max(H,W));image=cv2.resize(image,None,fx=scale,fy=scale) if scale<1 else image
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);h,w=gray.shape
    enhanced=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)).apply(gray)
    candidates=[];seen=[]
    for angle in [-60,-30,0,30,60,90]:
        matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
        c,s=abs(matrix[0,0]),abs(matrix[0,1]);rw,rh=int(h*s+w*c),int(h*c+w*s)
        matrix[0,2]+=rw/2-w/2;matrix[1,2]+=rh/2-h/2
        inv=cv2.invertAffineTransform(matrix)
        rotated=cv2.warpAffine(enhanced,matrix,(rw,rh),borderMode=cv2.BORDER_REPLICATE)
        gradient=np.abs(cv2.Sobel(rotated,cv2.CV_32F,1,0,ksize=3))
        gradient=np.clip(gradient,0,255).astype(np.uint8)
        for threshold in [cv2.threshold(gradient,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],(gradient>30).astype(np.uint8)*255]:
            close=cv2.morphologyEx(threshold,cv2.MORPH_CLOSE,np.ones((5,19),np.uint8))
            close=cv2.morphologyEx(close,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            n,_,stats,_=cv2.connectedComponentsWithStats(close,8)
            boxes=[]
            for x,y,bw,bh,area in stats[1:]:
                if bw>20 and bh>8 and bw/bh>1.1 and bw*bh>.00015*w*h and bw*bh<.12*w*h:
                    boxes.append((x,y,bw,bh))
            proposals=[]
            for x,y,bw,bh in boxes:
                if 2.35<bw/bh<6:proposals.append(((x-bh*.12,y-bh*.15,bw+bh*.24,bh*1.30),'single_row'))
                if 1.0<bw/bh<2.3:proposals.append(((x-bh*.05,y-bh*.05,bw+bh*.1,bh*1.1),'double_row'))
                for x2,y2,bw2,bh2 in boxes:
                    dy=y2-y
                    if not(.7*bh<dy<2.5*bh and .5<bh2/bh<1.8):continue
                    overlap=min(x+bw,x2+bw2)-max(x,x2)
                    if overlap<.6*min(bw,bw2):continue
                    lo=np.array([min(x,x2),min(y,y2)],float);hi=np.array([max(x+bw,x2+bw2),max(y+bh,y2+bh2)],float)
                    wh=hi-lo
                    if not 1.0<wh[0]/wh[1]<2.3:continue
                    pad=.12*min(bh,bh2);lo-=pad;hi+=pad
                    proposals.append(((*lo,*(hi-lo)),'double_row'))
            for (x,y,bw,bh),typ in proposals:
                if bw*bh>.12*w*h:continue
                q=np.float32([[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]])
                q=cv2.transform(q[None],inv)[0]
                if np.any(q[:,0]<0) or np.any(q[:,1]<0) or np.any(q[:,0]>=w) or np.any(q[:,1]>=h):continue
                q=order_quad_points(q);center=q.mean(axis=0)
                if any(np.linalg.norm(center-old)<.10*bw for old in seen):continue
                seen.append(center)
                ar=bw/bh;tw=int(ar*120);dst=np.float32([[0,0],[tw-1,0],[tw-1,119],[0,119]])
                flat=cv2.warpPerspective(gray,cv2.getPerspectiveTransform(q,dst),(tw,120))
                good=ink_layout(flat,typ)
                if not good:continue
                oq=q/scale;lo=oq.min(axis=0);hi=oq.max(axis=0)
                candidates.append({'quad':oq,'bbox':tuple(map(int,[*lo,*(hi-lo)])),'type':typ,'evidence':good,'score':good['score']})
    return sorted(candidates,key=lambda c:-c['score'])

