"""Create seven independent, reproducible image degradations per original.

No OCR guesses become ground truth. ROI hints only locate local effects.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,random
from collections import Counter

import cv2
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont

CATEGORIES=('01_mo','02_xien_duoi_45','03_lam_sang','04_lam_toi',
            '05_loa_sang','06_xoay_45_90','07_ban')
TITLES=('Blur','Perspective / tilt <45 deg','Brighter','Darker','Glare','Rotation 45-90 deg','Dirt')
EXTENSIONS={'.jpg','.jpeg','.png','.webp','.bmp'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_image(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()


def corners(width,height):
    return np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])


def project(points,matrix):
    return cv2.perspectiveTransform(np.asarray(points,dtype=np.float32).reshape(1,-1,2),matrix)[0]


def rotate_expand(image,angle):
    h,w=image.shape[:2]
    matrix=np.vstack([cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),angle,1.),[0.,0.,1.]])
    points=project(corners(w,h),matrix)
    low=np.floor(points.min(axis=0));high=np.ceil(points.max(axis=0))
    matrix[0,2]+=2-low[0];matrix[1,2]+=2-low[1]
    size=tuple((high-low+5).astype(int))
    return cv2.warpPerspective(image,matrix,size,flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=(55,55,55)),matrix


def clamp_box(box,w,h):
    x,y,bw,bh=map(float,box)
    x0=max(0,min(w-1,round(x)));y0=max(0,min(h-1,round(y)))
    x1=max(x0+1,min(w,round(x+bw)));y1=max(y0+1,min(h,round(y+bh)))
    return [x0,y0,x1-x0,y1-y0]


def load_hints(path,files):
    raw=json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    if 'records' in raw:
        hints={}
        for record in raw['records']:
            if record.get('orig_w',0)>0 and record.get('orig_h',0)>0:
                hints.setdefault(record['image_name'],[]).append([record['orig_'+axis] for axis in ('x','y','w','h')])
    else:hints=raw
    # Visually located plate regions for three original photos without prior
    # detector hits. These remain approximate effect targets, never GT labels.
    manual={
        '1793454835397':[[.565,.449,.040,.019]],
        '1793454835407':[[.346,.290,.046,.013],[.499,.296,.025,.017]],
        '1793454835416':[[.430,.495,.026,.012],[.707,.540,.036,.019]],
    }
    for file in files:
        if not hints.get(file.name) and file.stem[:13] in manual:
            image=read_image(file);h,w=image.shape[:2]
            hints[file.name]=[[x*w,y*h,bw*w,bh*h] for x,y,bw,bh in manual[file.stem[:13]]]
    return hints


def local_glare(image,boxes,rng):
    out=image.copy();details=[]
    for x,y,w,h in boxes:
        # Compute only around the plate; no full-resolution float canvas needed.
        left=max(0,x-w//3);top=max(0,y-h//2)
        right=min(image.shape[1],x+w+w//3);bottom=min(image.shape[0],y+h+h//2)
        yy,xx=np.mgrid[top:bottom,left:right].astype(np.float32)
        cx=x+w*rng.uniform(.30,.70);cy=y+h*rng.uniform(.35,.65)
        sx=max(2,w*rng.uniform(.17,.24));sy=max(2,h*rng.uniform(.25,.38))
        field=np.exp(-.5*(((xx-cx)/sx)**2+((yy-cy)/sy)**2))
        opacity=rng.uniform(.88,.98)*field
        patch=out[top:bottom,left:right].astype(np.float32)
        # A white veil removes local contrast, representing saturated reflection.
        patch=patch*(1-opacity[:,:,None])+np.array([255,253,242])*opacity[:,:,None]
        out[top:bottom,left:right]=np.clip(patch,0,255).astype(np.uint8)
        details.append({'center':[cx,cy],'sigma':[sx,sy],'peak_opacity':float(opacity.max())})
    return out,details


def local_dirt(image,boxes,rng):
    out=image.copy();details=[]
    for x,y,w,h in boxes:
        mask=np.zeros((h,w),np.uint8)
        for _ in range(rng.randint(9,15)):
            center=(rng.randrange(w),rng.randrange(h))
            axes=(max(1,round(w*rng.uniform(.018,.055))),max(1,round(h*rng.uniform(.025,.085))))
            cv2.ellipse(mask,center,axes,rng.randrange(180),0,360,rng.randint(100,220),-1)
        # Small specks, rather than a large occluder hiding the whole plate.
        for _ in range(rng.randint(25,45)):
            cv2.circle(mask,(rng.randrange(w),rng.randrange(h)),max(1,round(min(w,h)*.006)),rng.randint(70,150),-1)
        mask=cv2.GaussianBlur(mask,(3,3),.65)
        opacity=mask.astype(np.float32)/255
        color=np.array([rng.randrange(65,100),rng.randrange(45,70),rng.randrange(22,43)])
        patch=out[y:y+h,x:x+w].astype(np.float32)
        out[y:y+h,x:x+w]=np.clip(patch*(1-opacity[:,:,None])+color*opacity[:,:,None],0,255).astype(np.uint8)
        details.append({'coverage_fraction':float(np.mean(mask>30)),'color_rgb':color.tolist()})
    return out,details


def transform(image,kind,boxes,rng,index):
    h,w=image.shape[:2];identity=np.eye(3,dtype=np.float64)
    if kind=='01_mo':
        sigma=rng.uniform(.0018,.0028)*min(w,h)
        return cv2.GaussianBlur(image,(0,0),sigma),identity,{'method':'GaussianBlur','sigma_px':sigma}
    if kind=='02_xien_duoi_45':
        source=corners(w,h)
        destination=np.float32([[w*.08,h*.025],[w*.97,h*.08],[w*.87,h*.97],[w*.025,h*.91]])
        if index%2:destination[:,0]=w-1-destination[:,0];destination=destination[[1,0,3,2]]
        perspective=cv2.getPerspectiveTransform(source,destination)
        warped=cv2.warpPerspective(image,perspective,(w,h),flags=cv2.INTER_CUBIC,borderValue=(55,55,55))
        angle=rng.uniform(10,40)*(-1 if index%2 else 1)
        out,rotation=rotate_expand(warped,angle)
        return out,rotation@perspective,{'method':'perspective_then_rotation','angle_deg':angle,'destination_quad':destination.tolist()}
    if kind=='03_lam_sang':
        gamma=rng.uniform(.65,.80);gain=rng.uniform(1.10,1.22)
        lut=np.clip((np.arange(256)/255.)**gamma*255*gain,0,255).astype(np.uint8)
        return cv2.LUT(image,lut),identity,{'method':'gamma_gain','gamma':gamma,'gain':gain}
    if kind=='04_lam_toi':
        gamma=rng.uniform(1.55,1.95);gain=rng.uniform(.58,.74)
        lut=np.clip((np.arange(256)/255.)**gamma*255*gain,0,255).astype(np.uint8)
        return cv2.LUT(image,lut),identity,{'method':'gamma_gain','gamma':gamma,'gain':gain}
    if kind=='05_loa_sang':
        out,details=local_glare(image,boxes,rng)
        return out,identity,{'method':'local_white_veil','patches':details}
    if kind=='06_xoay_45_90':
        angle=(45.,60.,75.,90.)[index%4]*(-1 if (index//4)%2 else 1)
        out,matrix=rotate_expand(image,angle)
        return out,matrix,{'method':'rotation_expanded_canvas','angle_deg':angle}
    if kind=='07_ban':
        out,details=local_dirt(image,boxes,rng)
        return out,identity,{'method':'local_mud_and_specks','patches':details}
    raise ValueError('Unknown category: '+kind)


def save_image(path,array):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(array).save(path,format='JPEG',quality=95,subsampling=0)


def create_previews(root,records,source):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    # A full-frame sheet for each condition, covering all original photos.
    for kind,title in zip(CATEGORIES,TITLES):
        subset=[r for r in records if r['category']==kind]
        sheet=Image.new('RGB',(8*190,4*245+35),'#e8edf2');draw=ImageDraw.Draw(sheet)
        draw.text((8,8),title,fill='black',font=font)
        for i,record in enumerate(subset):
            with Image.open(root/record['output_image']) as image:
                thumb=image.copy();thumb.thumbnail((182,211))
            x=(i%8)*190;y=(i//8)*245+35
            sheet.paste(thumb,(x+(190-thumb.width)//2,y))
            draw.text((x+5,y+215),f"Image {record['source_index']:02d}",fill='black',font=font)
        sheet.save(root/'previews'/(kind+'.jpg'),quality=90)
    # Original plus seven changes for one source, enlarged around the plate.
    selected=[r for r in records if r['source_index']==1]
    first=selected[0];original=read_image(source/first['source_file'])
    x,y,w,h=first['effect_roi_hints'][0]
    box=corners(w,h)+[x,y]
    tiles=[('Original',original,box)]
    for record,title in zip(selected,TITLES):
        transformed=project(box,np.array(record['homography_source_to_output']))
        tiles.append((title,read_image(root/record['output_image']),transformed))
    sheet=Image.new('RGB',(4*410,2*305),'#e8edf2');draw=ImageDraw.Draw(sheet)
    for i,(title,image,poly) in enumerate(tiles):
        low=np.floor(poly.min(axis=0)).astype(int);high=np.ceil(poly.max(axis=0)).astype(int)
        margin=max(5,round((high-low).max()*.08))
        crop=Image.fromarray(image).crop((max(0,low[0]-margin),max(0,low[1]-margin),min(image.shape[1],high[0]+margin),min(image.shape[0],high[1]+margin)))
        crop.thumbnail((398,267));x=(i%4)*410;y=(i//4)*305
        sheet.paste(crop,(x+(410-crop.width)//2,y+27));draw.text((x+7,y+5),title,font=font,fill='black')
    sheet.save(root/'previews'/'so_sanh_7_truong_hop.jpg',quality=95)


def generate(source,output,hints=None,seed=20261105,expected_count=32):
    source=Path(source).resolve();output=Path(output).resolve()
    if output==source or output.is_relative_to(source):raise ValueError('Output must be outside the original input folder')
    files=sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if len(files)!=expected_count:raise ValueError(f'Expected {expected_count} originals, found {len(files)}')
    if output.exists() and any(output.iterdir()):raise ValueError('Output is not empty; choose another folder to avoid overwriting')
    output.mkdir(parents=True,exist_ok=True);(output/'previews').mkdir()
    rois=load_hints(hints,files);records=[];originals=[]
    for i,path in enumerate(files,1):
        before=digest(path);image=read_image(path);h,w=image.shape[:2]
        boxes=[clamp_box(box,w,h) for box in rois.get(path.name,[])]
        if not boxes:raise ValueError('Missing effect target for '+path.name)
        originals.append({'source_index':i,'source_file':path.name,'source_sha256':before,'width':w,'height':h})
        for kind in CATEGORIES:
            key=f'{seed}|{before}|{kind}'
            variant_seed=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)
            out,matrix,parameters=transform(image,kind,boxes,random.Random(variant_seed),i-1)
            relative=f'{kind}/anh_{i:02d}__{kind}.jpg';target=output/relative
            save_image(target,out)
            records.append({'source_index':i,'source_file':path.name,'source_sha256':before,
                'source_group':before,'provenance':'augmentation_of_real_photo','is_independent_real_capture':False,
                'category':kind,'output_image':relative,'output_sha256':digest(target),'seed':variant_seed,
                'original_size':[w,h],'output_size':[out.shape[1],out.shape[0]],'parameters':parameters,
                'homography_source_to_output':matrix.tolist(),'effect_roi_hints':boxes,
                'roi_status':'approximate_effect_targets_not_ground_truth','ocr_label':None,'split':'unassigned'})
            del out
        if before!=digest(path):raise RuntimeError('An original was modified during processing')
        print(f'Processed {i}/{len(files)} originals ({i*7} variants)',flush=True)
    with (output/'manifest.jsonl').open('w',encoding='utf-8') as handle:
        for record in records:handle.write(json.dumps(record,ensure_ascii=False)+'\n')
    with (output/'index.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['source_index','source_file','source_group','category','output_image','split'])
        writer.writeheader()
        for record in records:writer.writerow({key:record[key] for key in writer.fieldnames})
    (output/'original_sources.json').write_text(json.dumps(originals,ensure_ascii=False,indent=2),encoding='utf-8')
    create_previews(output,records,source)
    summary={'original_count':len(files),'new_image_count':len(records),'categories':dict(Counter(r['category'] for r in records)),
        'seed':seed,'source_dir':str(source),'output_dir':str(output),'completed':True,
        'warning':'All descendants of the same original must remain in the same train/validation/test group. No automatic OCR labels.'}
    (output/'dataset.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Create 7 variants of each original photo without replacing originals')
    base=Path(__file__).resolve().parents[1]
    parser.add_argument('--source',type=Path,default=base/'input/images')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--roi-hints',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261105)
    args=parser.parse_args()
    print(json.dumps(generate(args.source,args.output,args.roi_hints,args.seed),ensure_ascii=False,indent=2))
