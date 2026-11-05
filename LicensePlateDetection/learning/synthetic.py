"""Deterministic labelled line images and lazy, train-only synthetic datasets."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from functools import lru_cache
import hashlib,io,json,os,random,threading

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageEnhance,ImageFilter
from filelock import FileLock

from .dataset import ALPHABET,clean_label

BASE=Path(__file__).resolve().parents[1]
DEFAULT_ROOT=BASE/'data/synthetic_ocr'
VERSION=2
DEFAULT_FONTS=('bahnschrift.ttf','arialbd.ttf','ARIALNB.TTF')


def _json(path,payload):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(payload,handle,ensure_ascii=False,indent=2);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


@lru_cache(maxsize=256)
def _font(path,size):return ImageFont.truetype(path,size)


def find_fonts(paths=None):
    candidates=[Path(p) for p in paths] if paths else [Path('C:/Windows/Fonts')/name for name in DEFAULT_FONTS]
    found=[]
    for path in candidates:
        if path.is_file():
            _font(str(path),48)
            found.append(path.resolve())
    if not found:raise ValueError('Không tìm thấy font. Cung cấp --font đường_dẫn_font.ttf')
    return found


def make_text(index,rng):
    letters=ALPHABET[10:]
    first=letters[(index//5)%len(letters)]
    style=rng.randrange(4)
    prefix=str(rng.randrange(10,100))+first
    if style==1:prefix+=str(rng.randrange(10))
    elif style in (2,3):
        prefix+=rng.choice(letters)
        if style==3:prefix+=str(rng.randrange(10))
    serial=''.join(str(rng.randrange(10)) for _ in range(rng.choice((4,5,5))))
    shown_prefix=prefix[:2]+('-' if rng.random()<.7 else '')+prefix[2:]
    shown_serial=serial[:3]+'.'+serial[3:] if len(serial)==5 and rng.random()<.8 else serial
    kind=('prefix','prefix','serial','serial','single')[index%5]
    shown=shown_prefix if kind=='prefix' else (shown_serial if kind=='serial' else shown_prefix+rng.choice(('-', ' '))+shown_serial)
    return {'text':clean_label(shown),'rendered_text':shown,'kind':kind,'plate_id':prefix+serial}


def render_line(shown,font_path,rng):
    size=rng.randrange(42,69);stroke=rng.choice((0,0,0,1));font=_font(str(font_path),size)
    tracking=rng.uniform(1.,3.)
    boxes=[font.getbbox(ch,stroke_width=stroke) for ch in shown]
    advances=[font.getlength(ch)+tracking for ch in shown]
    box=(min(b[0] for b in boxes),min(b[1] for b in boxes),
         int(np.ceil(sum(advances[:-1])+boxes[-1][2])),max(b[3] for b in boxes))
    pad=rng.randrange(7,16)
    w,h=box[2]-box[0]+2*pad,box[3]-box[1]+2*pad
    palette=rng.randrange(12)
    if palette==0:bg=(35,70,130);fg=(235,240,245)
    elif palette in (1,2):bg=(rng.randrange(195,240),rng.randrange(175,220),rng.randrange(75,140));fg=(20,23,25)
    else:
        base=rng.randrange(165,251);bg=(base,max(0,base-rng.randrange(0,13)),max(0,base-rng.randrange(0,20)));fg=(rng.randrange(5,65),)*3
    image=Image.new('RGB',(w,h),bg)
    draw=ImageDraw.Draw(image);x=pad-box[0]
    for ch,advance in zip(shown,advances):
        draw.text((x,pad-box[1]),ch,font=font,fill=fg,stroke_width=stroke,stroke_fill=fg)
        x+=advance
    if rng.random()<.15:ImageDraw.Draw(image).line((0,h-2,w,h-2),fill=fg,width=1)
    # Expanded canvas retains all glyphs even at the most tilted settings.
    angle=rng.uniform(-15,15) if rng.random()<.18 else rng.uniform(-5,5)
    image=image.rotate(angle,Image.Resampling.BICUBIC,expand=True,fillcolor=bg)
    width=max(48,round(image.width*64/image.height))
    image=image.resize((width,64),Image.Resampling.LANCZOS)
    brightness=rng.uniform(.65,1.30)
    image=ImageEnhance.Brightness(image).enhance(brightness)
    image=ImageEnhance.Contrast(image).enhance(rng.uniform(.75,1.25))
    a=np.asarray(image,dtype=np.float32)
    shade=np.linspace(rng.uniform(.85,1.05),rng.uniform(.85,1.10),image.width,dtype=np.float32)
    a*=shade[None,:,None]
    noise=rng.uniform(0,7.)
    np_rng=np.random.default_rng(rng.getrandbits(64))
    a+=np_rng.normal(0,noise,a.shape)
    image=Image.fromarray(np.clip(a,0,255).astype(np.uint8))
    if rng.random()<.2:
        overlay=Image.new('RGBA',image.size)
        draw=ImageDraw.Draw(overlay)
        for _ in range(rng.choice((1,1,2))):
            x=rng.randrange(max(1,image.width//2));y=rng.randrange(6,58)
            draw.line((x,y,min(image.width-1,x+rng.randrange(12,max(13,image.width//2))),y+rng.randrange(-2,3)),fill=(*bg,90),width=1)
        image=Image.alpha_composite(image.convert('RGBA'),overlay).convert('RGB')
    blur=rng.uniform(0,1.35) if rng.random()<.65 else 0.
    if blur:image=image.filter(ImageFilter.GaussianBlur(blur))
    return image,{'angle':round(angle,3),'blur':round(blur,3),'noise':round(noise,3),'brightness':round(brightness,3),'stroke_width':stroke}


def generate_batch(root=DEFAULT_ROOT,count=20000,seed=20261105,fonts=None,exclude_plate_ids=(),cancel=None,progress=None):
    if not 1<=count<=100000:raise ValueError('Số ảnh mỗi đợt phải từ 1 đến 100.000')
    root=Path(root);font_paths=find_fonts(fonts);cancel=cancel or threading.Event();progress=progress or (lambda event:None)
    font_info=[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'path':str(p)} for p in font_paths]
    excluded=set(exclude_plate_ids)
    spec={'version':VERSION,'count':count,'seed':int(seed),'fonts':[{'name':p['name'],'sha256':p['sha256']} for p in font_info],
          'excluded_plate_ids':sorted(excluded)}
    batch_id=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:20]
    batch=root/'batches'/batch_id;batch.mkdir(parents=True,exist_ok=True)
    with FileLock(str(batch/'generation.lock'),timeout=0):
        metadata_path=batch/'dataset.json'
        if metadata_path.is_file():
            previous=json.loads(metadata_path.read_text(encoding='utf-8'))
            if previous.get('completed'):
                return {**previous,'batch_dir':str(batch),'reused':True}
        excluded=set(exclude_plate_ids);char_counts=Counter();kind_counts=Counter();preview=[];generated=0
        initial={**spec,'batch_id':batch_id,'source':'synthetic','split':'train','completed':False,
                 'fonts':font_info,'created_at':datetime.now(timezone.utc).isoformat()}
        _json(metadata_path,initial)
        temporary=batch/'manifest.jsonl.partial';labels_path=batch/'labels.tsv.partial'
        with temporary.open('w',encoding='utf-8') as manifest,labels_path.open('w',encoding='utf-8',newline='') as labels:
            for index in range(count):
                if cancel.is_set():break
                rng=random.Random(int(seed)^((index+1)*0x9e3779b97f4a7c15))
                text=make_text(index,rng)
                while text['plate_id'] in excluded:text=make_text(index,rng)
                font_path=font_paths[rng.randrange(len(font_paths))]
                image,effects=render_line(text['rendered_text'],font_path,rng)
                relative=f'images/{index//1000:04d}/{index:07d}.jpg'
                path=batch/relative;path.parent.mkdir(parents=True,exist_ok=True)
                image.save(path,format='JPEG',quality=rng.randrange(65,96),subsampling=0)
                record={'id':f'{batch_id}:{index}','image':relative,**text,'source':'synthetic','split':'train',
                        'font':font_path.name,'effects':effects,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                manifest.write(json.dumps(record,ensure_ascii=False)+'\n');labels.write(relative+'\t'+record['text']+'\n')
                char_counts.update(record['text']);kind_counts.update([record['kind']]);generated+=1
                if index<48:preview.append((image.copy(),record['text']))
                if generated%100==0 or generated==count:progress({'stage':'synthetic_generate','generated':generated,'total':count})
            manifest.flush();os.fsync(manifest.fileno());labels.flush();os.fsync(labels.fileno())
        completed=generated==count and not cancel.is_set()
        info={**initial,'completed':completed,'generated':generated,'character_counts':dict(char_counts),'kind_counts':dict(kind_counts),
              'excluded_plate_ids':sorted(excluded),'batch_dir':str(batch)}
        if completed:
            os.replace(temporary,batch/'manifest.jsonl');os.replace(labels_path,batch/'labels.tsv')
            info['manifest_sha256']=hashlib.sha256((batch/'manifest.jsonl').read_bytes()).hexdigest()
            sheet=Image.new('RGB',(1000,12*100),'#e8edf2');draw=ImageDraw.Draw(sheet)
            for i,(image,label) in enumerate(preview):
                image.thumbnail((238,74));x=(i%4)*250;y=(i//4)*100
                sheet.paste(image,(x+6,y+4));draw.text((x+6,y+80),label,fill='black')
            sheet.save(batch/'preview.jpg',quality=90)
        _json(metadata_path,info)
        return {**info,'reused':False}


class SyntheticStore:
    def __init__(self,root=DEFAULT_ROOT):self.root=Path(root)

    def batches(self):
        result=[]
        for path in sorted(self.root.glob('batches/*/dataset.json')):
            if not path.resolve().is_relative_to(self.root.resolve()):raise ValueError('Invalid synthetic dataset path')
            metadata=json.loads(path.read_text(encoding='utf-8'))
            if metadata.get('completed') and metadata.get('source')=='synthetic':
                result.append((path.parent,metadata))
        return result

    def count(self):return sum(m['generated'] for _,m in self.batches())

    def records(self,exclude_plate_ids=()):
        result=[];excluded=set(exclude_plate_ids)
        for root,metadata in self.batches():
            path=root/'manifest.jsonl'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['manifest_sha256']:raise ValueError('Manifest dữ liệu giả đã thay đổi')
            count=0
            with path.open(encoding='utf-8') as handle:
                for line in handle:
                    record=json.loads(line);count+=1
                    if record.get('source')!='synthetic' or record.get('split')!='train':raise ValueError('Ảnh giả chỉ được dùng cho tập train')
                    if clean_label(record['rendered_text'])!=record['text']:raise ValueError('Nhãn ảnh giả không khớp chuỗi đã vẽ')
                    if record['plate_id'] not in excluded:result.append({**record,'batch_root':str(root)})
            if count!=metadata['count']:raise ValueError('Lô ảnh giả chưa đủ số lượng trong manifest')
        return result

    def load(self,record):
        root=Path(record['batch_root']).resolve();path=(root/record['image']).resolve()
        if not root.is_relative_to(self.root.resolve()) or not path.is_relative_to(root):raise ValueError('Invalid synthetic image path')
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=record['sha256']:raise ValueError('Ảnh giả không còn khớp nhãn đã sinh')
        with Image.open(io.BytesIO(payload)) as image:return image.convert('RGB'),record['text']

    def digest(self):
        return hashlib.sha256(json.dumps([(m['batch_id'],m['manifest_sha256']) for _,m in self.batches()],sort_keys=True).encode()).hexdigest()
