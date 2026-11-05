"""A trainable VGG-BiLSTM-CTC reader initialized from EasyOCR english_g2.

Only network layers/weights are reused. Images remain local. Dataset handling,
row extraction, canonical alphabet, training and checkpoint selection are ours.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import threading

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

from .vendor.easyocr_layers import VGG_FeatureExtractor, BidirectionalLSTM

ALPHABET = '0123456789ABCDEFGHKLMNPSTUVXYZ'
BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE/'models/line_ocr'
_LOCK = threading.RLock()
_CACHE = {}


class CRNN(nn.Module):
    def __init__(self, classes=97):
        super().__init__()
        self.FeatureExtraction = VGG_FeatureExtractor(1, 256)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(256, 256, 256), BidirectionalLSTM(256, 256, 256))
        self.Prediction = nn.Linear(256, classes)

    def forward(self, images):
        visual = self.FeatureExtraction(images).permute(0, 3, 1, 2)
        visual = self.AdaptiveAvgPool(visual).squeeze(3)
        return self.Prediction(self.SequenceModeling(visual))


def canonical_log_probs(logits, charset):
    """Merge upper/lower case into 30 classes while retaining CTC blank 0."""
    # A visible dash/dot must become a separator, not a forced digit/letter.
    blank=[0]+[i+1 for i,raw in enumerate(charset) if not raw.isalnum()]
    groups = [blank] + [[i+1 for i, raw in enumerate(charset) if raw.upper()==char] for char in ALPHABET]
    if any(not group for group in groups):
        raise ValueError('Pretrained alphabet is missing a plate character')
    merged = torch.stack([torch.logsumexp(logits[..., group], dim=-1) for group in groups], dim=-1)
    return merged.log_softmax(-1)


def choose_device(device='auto'):
    if device=='auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('PyTorch hiện tại chưa hỗ trợ GPU; chọn CPU hoặc cài bản CUDA.')
    return torch.device(device)


def load_model(model_dir=MODEL_DIR, device='auto', checkpoint=None):
    model_dir=Path(model_dir)
    charset=json.loads((model_dir/'charset.json').read_text(encoding='utf-8'))
    selected=Path(checkpoint) if checkpoint else model_dir/'english_g2.pth'
    version='pretrained-english-g2'
    info={}
    pointer=model_dir/'active.json'
    if checkpoint is None and pointer.is_file():
        info=json.loads(pointer.read_text(encoding='utf-8'))
        selected=(model_dir/info['checkpoint']).resolve()
        if not selected.is_relative_to(model_dir.resolve()):
            raise ValueError('Checkpoint path outside model directory')
        version=info['version']
        if hashlib.sha256(selected.read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError('Trọng số đang dùng không khớp thông tin kiểm tra')
    payload=torch.load(selected,map_location='cpu',weights_only=True)
    if checkpoint is not None:
        version=payload.get('metadata',{}).get('version','pretrained-english-g2' if selected.name=='english_g2.pth' else 'checkpoint:'+selected.stem)
    weights=payload.get('state_dict',payload)
    weights={key.removeprefix('module.'):value for key,value in weights.items()}
    model=CRNN(len(charset)+1)
    model.load_state_dict(weights,strict=True)
    model._training_character_counts=info.get('trained_character_counts',payload.get('metadata',{}).get('trained_character_counts',{}))
    device=choose_device(device)
    model.to(device).eval()
    return model,charset,device,version


def prepare_batch(images, device='cpu', height=64, max_width=512):
    resized=[];widths=[]
    for image in images:
        image=image.convert('L') if isinstance(image,Image.Image) else Image.fromarray(image).convert('L')
        if min(image.size)<2:
            raise ValueError('Dòng ảnh quá nhỏ')
        width=max(16,min(max_width,math.ceil(height*image.width/image.height)))
        a=np.asarray(image.resize((width,height),Image.Resampling.BICUBIC),dtype=np.float32)/127.5-1.
        resized.append(a);widths.append(width)
    width=max(widths)
    batch=np.zeros((len(images),1,height,width),dtype=np.float32)
    for i,a in enumerate(resized):
        batch[i,0,:,:a.shape[1]]=a
        batch[i,0,:,a.shape[1]:]=a[:,-1:]
    return torch.from_numpy(batch).to(device),torch.tensor([w//4-1 for w in widths],dtype=torch.long),widths


def split_rows(image, plate_type='double_row', ratio=None):
    image=image.convert('RGB')
    w,h=image.size
    if plate_type=='single_row':
        return [(image,(0,0,w,h))],None
    if ratio is None:
        gray=np.asarray(image.convert('L'))
        gx=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0))
        profile=gx[:,max(1,round(w*.07)):max(2,round(w*.93))].mean(axis=1)
        profile=cv2.GaussianBlur(profile[:,None],(1,5),0).ravel()
        low,high=round(h*.40),round(h*.60)
        candidates=np.arange(low,max(low+1,high))
        penalties=profile[candidates]+.15*max(profile.max(),1)*np.abs(candidates/h-.5)
        ratio=float(candidates[np.argmin(penalties)]/h)
    if not .25<=ratio<=.75:
        raise ValueError('Vị trí tách dòng phải nằm trong 25–75% chiều cao biển')
    cut=round(h*ratio);bx=max(0,round(w*.025));by=max(0,round(h*.025))
    boxes=[(bx,by,w-bx,cut),(bx,cut,w-bx,h-by)]
    return [(image.crop(box),(box[0],box[1],box[2]-box[0],box[3]-box[1])) for box in boxes],ratio


def decode(log_probs, length, digits_only=False):
    probabilities=log_probs[:length].exp()
    allowed=list(range(11)) if digits_only else list(range(len(ALPHABET)+1))
    path=probabilities[:,allowed].argmax(-1)
    path=[allowed[int(i)] for i in path]
    chars=[];previous=0
    for t,index in enumerate(path):
        if index and index!=previous:
            score=float(probabilities[t,index])
            chars.append({'char':ALPHABET[index-1],'score':score,'start':t,'end':t+1})
        elif index and chars:
            chars[-1]['end']=t+1
            chars[-1]['score']=max(chars[-1]['score'],float(probabilities[t,index]))
        previous=index
    return chars


def predict_lines(model,charset,device,images,digits=None):
    tensors,lengths,_=prepare_batch(images,device)
    with torch.inference_mode():
        log_probs=canonical_log_probs(model(tensors),charset).cpu()
    result=[]
    for i,image in enumerate(images):
        n=min(int(lengths[i]),log_probs.shape[1])
        chars=decode(log_probs[i],n,bool(digits and digits[i]))
        text=''.join(c['char'] for c in chars)
        score=float(np.mean([c['score'] for c in chars])) if chars else 0.
        result.append({'text':text,'score':score,'characters':chars,'frames':n})
    return result


def engine(model_dir=MODEL_DIR):
    model_dir=Path(model_dir)
    pointer=model_dir/'active.json'
    token=(str(model_dir.resolve()),pointer.stat().st_mtime_ns if pointer.exists() else 0)
    with _LOCK:
        if token not in _CACHE:
            torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
            loaded=load_model(model_dir)
            _CACHE.clear();_CACHE[token]=loaded
        return _CACHE[token]


def recognize(image,plate_type='double_row',model_dir=MODEL_DIR):
    from image_processing.character_recognition import is_valid_vn_plate
    model,charset,device,version=engine(model_dir)
    results=[]
    with _LOCK:
        for angle in (0,180):
            oriented=image.convert('RGB').rotate(angle,expand=True) if angle else image.convert('RGB')
            rows,ratio=split_rows(oriented,plate_type)
            predictions=predict_lines(model,charset,device,[r[0] for r in rows],
                                      [False,True] if len(rows)==2 else [False])
            chars=[]
            for row_index,(pred,(_,box)) in enumerate(zip(predictions,rows),1):
                x,y,w,h=box
                for c in pred['characters']:
                    left=x+round(c['start']/pred['frames']*w)
                    right=x+round(c['end']/pred['frames']*w)
                    chars.append({'char':c['char'],'row':row_index,'score':round(c['score'],4),
                                  'bbox':(left,y,max(2,right-left),h),
                                  'bbox_source':'ctc_alignment',
                                  'match_details':{'backend':'crnn','char':c['char'],'score':round(c['score'],4),
                                                   'needs_review':c['score']<.8,'syntax_conflict':False,
                                                   'top_candidates':[{'char':c['char'],'score':round(c['score'],4)}]}})
            text=''.join(c['char'] for c in chars) or 'KXD'
            score=float(np.mean([c['score'] for c in chars])) if chars else 0.
            counts=[sum(c['row']==r for c in chars) for r in range(1,len(rows)+1)]
            complete=(3<=counts[0]<=5 and 4<=counts[1]<=5) if len(rows)==2 else 7<=counts[0]<=9
            seen=getattr(model,'_training_character_counts',{})
            unseen=sorted({c['char'] for c in chars if seen and c['char'] not in seen})
            for c in chars:
                if c['char'] in unseen:c['match_details']['needs_review']=True
            review=not complete or not is_valid_vn_plate(text) or any(c['score']<.8 for c in chars) or bool(unseen)
            reasons=[]
            if unseen:reasons.append('Chưa có ảnh gán nhãn trong tập học cho ký tự: '+', '.join(unseen))
            if not complete:reasons.append('Số ký tự chưa khớp bố cục một/hai dòng')
            if not is_valid_vn_plate(text):reasons.append('Chuỗi chưa khớp bố cục thông dụng; cần đối chiếu ảnh')
            if any(c['score']<.8 for c in chars):reasons.append('Có ký tự chưa đủ tin cậy')
            gray=np.asarray(oriented.convert('L'))
            _,foreground=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            covered=np.zeros_like(foreground,dtype=bool)
            for _,(x,y,w,h) in rows:
                covered[y:y+h,x:x+w]=True
            coverage=float(np.sum((foreground>0)&covered)/max(1,np.sum(foreground>0)))
            prefix_valid=bool(re.fullmatch(r'\d{2}[ABCDEFGHKLMNPSTUVXYZ]{1,2}\d?',predictions[0]['text'])) if len(rows)==2 else False
            if len(rows) == 2 and len(predictions) == 2:
                p1 = predictions[0]['text']
                p2 = predictions[1]['text']
                p1_fmt = f"{p1[:2]} {p1[2:]}" if p1[:2].isdigit() and len(p1) > 2 else p1
                p2_fmt = f"{p2[:3]}.{p2[3:]}" if len(p2) == 5 else p2
                disp_text = f"{p1_fmt} {p2_fmt}".strip() if (p1 or p2) else text
            else:
                from image_processing.character_recognition import format_plate_display
                disp_text = format_plate_display(text, plate_type)
            results.append({'plate_text':disp_text,'char_count':len(chars),'avg_confidence':round(score,3),
                            'characters':chars,'recognizer':'crnn-english-g2:'+version,'needs_review':review,
                            'uncertain_positions':[i for i,c in enumerate(chars) if c['match_details']['needs_review']],
                            'uncertainty_reasons':reasons,
                            'deskewed_image':oriented,'ink_coverage':coverage,'ink_coverage_source':'row_regions',
                            'row_split_ratio':ratio,'line_predictions':[{k:v for k,v in p.items() if k!='characters'} for p in predictions],
                            '_quality':score+.25*complete+.30*is_valid_vn_plate(text)+.04*text[:2].isdigit()+.15*prefix_valid})
    best=results[1] if results[1]['_quality']>results[0]['_quality']+.06 else results[0]
    best.pop('_quality',None)
    return best
