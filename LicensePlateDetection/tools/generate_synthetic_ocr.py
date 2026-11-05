"""Generate local, deterministic line images for the project's CRNN."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    from learning.synthetic import generate_batch,DEFAULT_ROOT
    from learning.dataset import DatasetStore
    parser=argparse.ArgumentParser(description='Sinh ảnh dòng biển số kèm nhãn chuẩn theo chuỗi đã vẽ')
    parser.add_argument('--count',type=int,default=20000)
    parser.add_argument('--seed',type=int,default=20261105)
    parser.add_argument('--output',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--font',action='append',help='Font .ttf; có thể lặp lại tùy chọn này')
    args=parser.parse_args()
    excluded={r['plate_id'] for r in DatasetStore().list_examples() if r['split']!='train'}
    def progress(event):
        if event['generated']%1000==0 or event['generated']==event['total']:
            print(f"Generated {event['generated']}/{event['total']}",flush=True)
    result=generate_batch(args.output,args.count,args.seed,args.font,excluded,progress=progress)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
