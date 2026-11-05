"""Fine-tune the line OCR on explicitly labelled local images."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser(description='Huấn luyện bổ sung CRNN từ ảnh biển đã gán nhãn')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--learning-rate',type=float,default=.0001)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--synthetic',action='store_true',help='Trộn ảnh giả với ảnh thật')
    parser.add_argument('--synthetic-root',type=Path)
    parser.add_argument('--synthetic-per-epoch',type=int,default=256,help='0: toàn bộ ảnh giả mỗi vòng; mặc định 256 để thử trên CPU')
    initial=parser.add_mutually_exclusive_group()
    initial.add_argument('--from-pretrained',action='store_true')
    initial.add_argument('--start-checkpoint',type=Path)
    args=parser.parse_args()
    from learning.trainer import train
    def show(event):
        if event['stage']=='epoch':
            print(json.dumps({'epoch':event['epoch'],'loss':event['loss'],
                  'validation_CER':event['validation']['character_error_rate'],
                  'validation_correct':event['validation']['correct'],
                  'training_CER':event['training']['character_error_rate']},ensure_ascii=False),flush=True)
        else:print(event['stage'],flush=True)
    from learning.line_model import MODEL_DIR
    checkpoint=MODEL_DIR/'english_g2.pth' if args.from_pretrained else args.start_checkpoint
    report=train(epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,device=args.device,progress=show,
                 use_synthetic=args.synthetic,synthetic_root=args.synthetic_root,synthetic_per_epoch=args.synthetic_per_epoch,
                 start_checkpoint=checkpoint)
    print(json.dumps({'activated':report['activated'],'report':str(Path(report['run_dir'])/'report.json')},ensure_ascii=False))

if __name__=='__main__':main()
