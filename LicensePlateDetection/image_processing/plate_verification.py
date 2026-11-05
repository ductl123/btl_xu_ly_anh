import cv2
import numpy as np
from utils.image_utils import order_quad_points
from .object_verifier import object_margin

def ink_layout(gray,plate_type):
    h,w=gray.shape
    methods=[cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1],
             cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,7)]
    methods += [255-mask for mask in list(methods)]
    best=None
    from .character_recognition import _strip_plate_frame
    methods += [_strip_plate_frame(mask>0).astype(np.uint8)*255 for mask in list(methods)]
    for mask in methods:
        mask=mask.copy();mask[:5]=0;mask[-5:]=0;mask[:,:5]=0;mask[:,-5:]=0
        # Repair sub-pixel breaks after crop normalization, without bridging rows.
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2,1),np.uint8))
        if not .03 < np.mean(mask > 0) < .60:
            continue
        n,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
        chars=[]
        for st,ct in zip(stats[1:],centers[1:]):
            x,y,cw,ch,area=st
            if .14*h<ch<.90*h and 2<=cw<.20*w and .06<cw/ch<1.25 and .12<area/(cw*ch)<.98:
                chars.append((x,y,cw,ch,*ct))
        if len(chars)<5 or len(chars)>15:continue
        heights=np.array([c[3] for c in chars]);mh=np.median(heights)
        chars=[c for c in chars if .55*mh<c[3]<1.5*mh]
        if len(chars)<5:continue
        ysort=sorted(chars,key=lambda c:c[5]);cys=np.array([c[5] for c in ysort])
        if plate_type=='double_row':
            split=int(np.argmax(np.diff(cys)))+1
            if cys[split]-cys[split-1]<.45*mh:continue
            rows=[ysort[:split],ysort[split:]]
            if not(2<=len(rows[0])<=5 and 3<=len(rows[1])<=6):continue
        else:
            rows=[ysort]
            if not 6<=len(chars)<=10:continue
        widths=[];align=[];cover=[]
        for row in rows:
            rh=np.median([c[3] for c in row]);cy=np.array([c[5] for c in row]);cx=np.array([c[4] for c in row])
            align.append(float(np.std(cy)/max(rh,1)))
            widths.append(np.ptp(cx)/w)
            cover.append(sum(c[2] for c in row)/w)
        if max(align)>.25 or min(cover)<.18:continue
        if plate_type=='double_row' and (widths[0]<.20 or widths[1]<.32):continue
        if plate_type=='single_row' and widths[0]<.40:continue
        if plate_type=='double_row' and not .55 < np.median([c[3] for c in rows[0]])/np.median([c[3] for c in rows[1]]) < 1.6:continue
        score=.55*min(len(chars)/8,1)+.25*(1-min(max(align)/.25,1))+.20*min(min(widths)/.65,1)
        evidence={'score':score,'counts':[len(row) for row in rows],'alignment':max(align),'foreground':float(np.mean(mask>0))}
        if best is None or score>best['score']:best=evidence
    return best

def recognized_row_evidence(result, cropped, plate_type):
    """Secondary check after removing connected frames and separating pairs."""
    if not complete_character_layout(result,plate_type) or result.get('avg_confidence',0)<.75:
        return None
    chars=result.get('characters',[])
    if not chars or min(c['score'] for c in chars)<.58 or result.get('ink_coverage',0)<.55:
        return None
    width,height=cropped.size
    row_groups=[[c for c in chars if c['row']==r] for r in ([1,2] if plate_type=='double_row' else [1])]
    aligns=[]
    for row in row_groups:
        heights=np.array([c['bbox'][3] for c in row]);centers=np.array([[c['bbox'][0]+c['bbox'][2]/2,c['bbox'][1]+c['bbox'][3]/2] for c in row])
        if np.std(centers[:,1])/max(1.,np.median(heights))>.22 or np.ptp(centers[:,0])<.20*width:
            return None
        aligns.append(float(np.std(centers[:,1])/max(1.,np.median(heights))))
    return {'score':.85,'counts':[len(row) for row in row_groups],
            'alignment':max(aligns),'foreground':.2,'source':'repaired_component_rows'}


def detect_boundaries(image,max_dim=1800):
    H,W=image.shape[:2];scale=min(1,max_dim/max(H,W))
    a=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else image
    h,w=a.shape[:2];g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);blur=cv2.GaussianBlur(g,(3,3),.7)
    hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV);sat=hsv[:,:,1];val=hsv[:,:,2];hue=hsv[:,:,0]
    neutral=(sat<85)|((hue>15)&(hue<45))
    masks=[cv2.morphologyEx(cv2.Canny(blur,lo,hi),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)) for lo,hi in [(12,40),(25,80),(60,160)]]
    for t in [60,90,120,150,180,210]:
        mask=((g>t)&neutral).astype(np.uint8)*255
        masks.append(cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)))
    candidates=[];seen=[]
    for mask_index, mask in enumerate(masks):
        contours=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
        for cnt in contours:
            contour_area=cv2.contourArea(cnt)
            hull=cv2.convexHull(cnt);ha=cv2.contourArea(hull)
            if ha<=0:continue
            area=ha if mask_index<3 else contour_area
            if not .00015*w*h<area<.15*w*h:continue
            if contour_area/ha < (.10 if mask_index<3 else .72):continue
            perimeter=cv2.arcLength(hull,True)
            for eps in [.015,.025,.04]:
                poly=cv2.approxPolyDP(hull,eps*perimeter,True)
                if len(poly)!=4 or not cv2.isContourConvex(poly):
                    rect=cv2.minAreaRect(hull)
                    if rect[1][0]*rect[1][1]<=0 or ha/(rect[1][0]*rect[1][1])<.80:
                        continue
                    poly=cv2.boxPoints(rect).reshape(4,1,2)
                q=order_quad_points(poly.reshape(4,2));qw=max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3]));qh=max(np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1]))
                if qh<12 or qw<30:continue
                ar=qw/qh
                if not 1.0<=ar<=6.:
                    continue
                previous=np.roll(q,1,axis=0)-q;following=np.roll(q,-1,axis=0)-q
                angles=np.degrees(np.arccos(np.clip((previous*following).sum(axis=1)/(np.linalg.norm(previous,axis=1)*np.linalg.norm(following,axis=1)),-1,1)))
                if angles.min()<40 or angles.max()>140:continue
                center=q.mean(axis=0)
                if any(np.linalg.norm(center-c)<.04*qw and abs(area-ar0)/area<.10 for c,ar0 in seen):continue
                seen.append((center,area))
                # Under strong foreshortening, the physical width can project
                # shorter than height. Resolve axes by text rows, not edge length.
                hypotheses=[]
                if ar<2.8:
                    hypotheses.append((q,'double_row',ar,None))
                if ar>=2.2:
                    hypotheses.append((q,'single_row',ar,None))
                if ar>=1.0:
                    hypotheses.append((np.roll(q,-1,axis=0),'double_row',1.4,1.4))
                for ordered,typ,target_ar,forced_ar in hypotheses:
                    tw,th=int(target_ar*120),120
                    dst=np.float32([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]])
                    M=cv2.getPerspectiveTransform(ordered.astype(np.float32),dst)
                    flat=cv2.warpPerspective(g,M,(tw,th))
                    color=cv2.warpPerspective(neutral.astype(np.uint8),M,(tw,th))
                    if color[6:-6,6:-6].mean()<.65:continue
                    good=ink_layout(flat,typ)
                    margin=object_margin(flat)
                    if margin is not None and margin < -.15:
                        continue
                    if good is None:
                        p10,p90=np.percentile(flat,(10,90))
                        if forced_ar is not None or margin is None or margin<.72 or p10<70 or p90-p10>130 or flat.mean()<100:
                            continue
                        good={'score':.84,'counts':[], 'alignment':0.,'foreground':.2,
                              'localization_only':True,'appearance':'weak_ink'}
                    good['object_margin']=margin
                    origq=ordered/scale;lo=origq.min(axis=0);hi=origq.max(axis=0)
                    candidates.append({'quad':origq,'bbox':tuple(map(int,[*lo,*(hi-lo)])),
                                       'type':typ,'evidence':good,'score':good['score'],
                                       'ordered_quad':True,'target_aspect_ratio':forced_ar})
    return sorted(candidates,key=lambda c:-c['score'])


def plate_evidence(cropped, plate_type):
    """Class-independent character rows, before trusting a formatted OCR string."""
    image=np.array(cropped.convert('RGB'))
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    if min(gray.shape)<8:
        return None
    gray=cv2.resize(gray,(max(20,int(gray.shape[1]*120/gray.shape[0])),120))
    return ink_layout(gray,plate_type)


def boundary_candidates(image, detection_scale):
    result=[]
    proposals=detect_boundaries(image)
    if not proposals:
        from .rotated_search import rotated_proposals
        proposals=rotated_proposals(image)
    for idx, candidate in enumerate(proposals):
        quad=candidate['quad']*detection_scale
        lo=np.floor(quad.min(axis=0)).astype(int)
        hi=np.ceil(quad.max(axis=0)).astype(int)
        width,height=(hi-lo).tolist()
        edge=quad[1]-quad[0]
        physical_width=max(np.linalg.norm(quad[1]-quad[0]),np.linalg.norm(quad[2]-quad[3]))
        physical_height=max(np.linalg.norm(quad[3]-quad[0]),np.linalg.norm(quad[2]-quad[1]))
        result.append({'id':-idx-1,'bbox':(int(lo[0]),int(lo[1]),width,height),
                       'quad_corners':quad,'boundary_refined':True,'source':'boundary',
                       'ordered_quad':candidate.get('ordered_quad',False),
                       'target_aspect_ratio':candidate.get('target_aspect_ratio'),
                       'angle':float(np.degrees(np.arctan2(edge[1],edge[0]))),
                       'plate_type':candidate['type'],'score':float(candidate['score']),
                       'aspect_ratio':float(candidate.get('target_aspect_ratio') or physical_width/physical_height),
                       'density':candidate['evidence']['foreground'],
                       'area':int(width*height*candidate['evidence']['foreground']),
                       'bbox_area':width*height,'layout_evidence':candidate['evidence']})
    return result


def deduplicate_candidates(candidates, limit=16):
    from .candidate_filter import compute_iou_and_ios
    ranked=sorted(candidates,key=lambda c:c.get('score',0)+(.5 if c.get('source')=='boundary' else 0),reverse=True)
    kept=[]
    for candidate in ranked:
        if any((lambda overlap: overlap[0]>.35 or overlap[1]>.8)(compute_iou_and_ios(candidate['bbox'],previous['bbox'])) for previous in kept):
            continue
        kept.append(candidate)
        if len(kept)>=limit:
            break
    return kept


def complete_character_layout(result, plate_type):
    chars=result.get('characters',[])
    if plate_type=='double_row':
        top=[c for c in chars if c['row']==1]
        bottom=[c for c in chars if c['row']==2]
        return 3<=len(top)<=5 and 4<=len(bottom)<=5
    return 7<=len(chars)<=9
