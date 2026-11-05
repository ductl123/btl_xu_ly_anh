"""Cut annotation crops from archive originals; do not infer OCR labels."""
from pathlib import Path
from collections import Counter
import csv,json,hashlib,argparse,math
from PIL import Image,ImageDraw,ImageFont


def prepare(manifest,out):
    rows=json.loads(Path(manifest).read_text(encoding='utf-8'));out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    accepted=[];skipped=[];seen=set()
    for source in rows:
        if source['augmented']:continue
        if source['sha256'] in seen:continue
        seen.add(source['sha256'])
        with Image.open(source['image']) as image:
            image=image.convert('RGB');w,h=image.size
            split={'train':'train','val':'validation','test':'test'}[source['split']]
            group=hashlib.sha256(('archive:'+source['source_group']).encode()).hexdigest()
            for i,(_,x,y,bw,bh) in enumerate(source['boxes'],1):
                if bw*w<24 or bh*h<14:
                    skipped.append({'source':Path(source['image']).name,'box':i,'reason':'plate too small for reliable transcription'})
                    continue
                box=(max(0,math.floor((x-bw*.54)*w)),max(0,math.floor((y-bh*.55)*h)),
                     min(w,math.ceil((x+bw*.54)*w)),min(h,math.ceil((y+bh*.55)*h)))
                crop=image.crop(box)
                if min(crop.size)<20:
                    factor=20/min(crop.size);crop=crop.resize(tuple(math.ceil(a*factor) for a in crop.size),Image.Resampling.BICUBIC)
                name=f"{Path(source['image']).stem}__bien_{i:02d}.png"
                relative=Path('crops')/split/name;target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
                crop.save(target)
                accepted.append({'image':relative.as_posix(),'row1':'','row2':'','split':split,'split_ratio':'',
                                 'source_group':group,'original_file':Path(source['image']).name,'box_index':i,
                                 'source_sha256':source['sha256'],'label_status':'needs_human_transcription'})
    fields=['image','row1','row2','split','split_ratio','source_group','original_file','box_index','source_sha256','label_status']
    with (out/'cho_gan_nhan_ocr.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(accepted)
    summary={'crops':len(accepted),'counts_by_split':dict(Counter(r['split'] for r in accepted)),
             'unique_source_images':len(seen),'too_small_for_transcription':len(skipped),'labelled_ocr_examples':0,
             'augmented_copies_excluded_from_initial_annotation_queue':sum(r['augmented'] for r in rows),
             'note':'Only detection box annotations were used. OCR text fields are intentionally blank.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'too_small.json').write_text(json.dumps(skipped,ensure_ascii=False,indent=2),encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    sheet=Image.new('RGB',(6*250,4*155),'#e8edf2');draw=ImageDraw.Draw(sheet)
    chosen=[accepted[round(i*(len(accepted)-1)/23)] for i in range(24)]
    for i,row in enumerate(chosen):
        with Image.open(out/row['image']) as image:
            thumb=image.copy();thumb.thumbnail((240,122))
        x=i%6*250;y=i//6*155;sheet.paste(thumb,(x+5,y+4))
        draw.text((x+5,y+129),row['split']+' | '+row['original_file'][:16],font=font,fill='black')
    sheet.save(out/'xem_mau_vung_bien.jpg',quality=93)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.manifest,a.output)
