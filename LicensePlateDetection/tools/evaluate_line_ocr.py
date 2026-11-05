"""Evaluate without updating weights or selecting a checkpoint."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['validation','test'],default='test')
    args=parser.parse_args()
    from learning.dataset import DatasetStore
    from learning.line_model import load_model,MODEL_DIR
    from learning.trainer import evaluate,atomic_json
    store=DatasetStore();records=store.list_examples(args.split)
    if not records:raise SystemExit('Chưa có ảnh cho tập '+args.split)
    model,chars,device,version=load_model()
    report={'version':version,'split':args.split,**evaluate(model,chars,device,store,records)}
    atomic_json(MODEL_DIR/('last_'+args.split+'_evaluation.json'),report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
