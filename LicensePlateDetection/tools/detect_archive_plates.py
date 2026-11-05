"""Run the archive detector on an image or folder, saving boxes and crops."""
from pathlib import Path
from datetime import datetime
import argparse,json,sys,os
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import torch

BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
try:from learning.archive_detector import load_detector,predict
except ModuleNotFoundError:from archive_detector import load_detector,predict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,nargs='?',help='Image or folder of images; opens a file picker if omitted')
    parser.add_argument('--model',type=Path,default=BASE/'models/archive_detector/best.pth')
    parser.add_argument('--output',type=Path,default=BASE/'output'/('archive_detector_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--score',type=float,default=None,help='Use validation-selected threshold by default')
    args=parser.parse_args();torch.set_num_threads(4)
    interactive=args.input is None
    if interactive:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk();root.withdraw()
        try:path=filedialog.askopenfilename(title='Chọn ảnh để tìm biển số',filetypes=[('Ảnh','*.jpg *.jpeg *.png *.webp *.bmp')])
        finally:root.destroy()
        if not path:return
        args.input=Path(path)
    model,checkpoint=load_detector(args.model);threshold=checkpoint['threshold'] if args.score is None else args.score
    if not 0<threshold<=1:raise ValueError('Score threshold must be in (0, 1]')
    supported={'.jpg','.jpeg','.png','.webp','.bmp'}
    images=sorted(p for p in args.input.iterdir() if p.suffix.lower() in supported) if args.input.is_dir() else [args.input]
    if not images:raise ValueError('No image files found')
    if args.output.resolve()==args.input.resolve():raise ValueError('Select an output folder separate from input')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'crops').mkdir()
    summary=[]
    for index,path in enumerate(images,1):
        with Image.open(path) as source:image=ImageOps.exif_transpose(source).convert('RGB')
        found=predict(model,np.asarray(image),threshold);preview=image.copy();draw=ImageDraw.Draw(preview)
        rows=[];w,h=image.size
        for n,(x0,y0,x1,y1,score) in enumerate(found,1):
            coords=[round(float(x0)),round(float(y0)),round(float(x1)),round(float(y1))]
            draw.rectangle(coords,outline='#23fa50',width=max(2,round(max(w,h)/480)))
            draw.text((coords[0],max(0,coords[1]-14)),f'{n}: {score:.2f}',fill='#ffffff',stroke_width=2,stroke_fill='#1a1a1a')
            mx=(x1-x0)*.045;my=(y1-y0)*.075
            box=(max(0,int(x0-mx)),max(0,int(y0-my)),min(w,int(x1+mx)+1),min(h,int(y1+my)+1))
            relative=f'crops/{index:04d}_{path.stem}_bien_{n}.png';image.crop(box).save(args.output/relative)
            rows.append({'bbox_xyxy':coords,'score':float(score),'crop':relative})
        preview.save(args.output/f'{index:04d}_{path.stem}_phat_hien.jpg',quality=94)
        summary.append({'input_image':str(path.resolve()),'detections':rows})
        print(f'{path.name}: {len(rows)} plates',flush=True)
    (args.output/'detections.json').write_text(json.dumps({'threshold':threshold,'images':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved to',args.output)
    if interactive:os.startfile(str(args.output.resolve()))


if __name__=='__main__':main()
