"""Import human labels from a CSV, never from predicted OCR strings."""
from pathlib import Path
import argparse,csv,hashlib,sys
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('csv_path',type=Path,help='CSV columns: image,row1,row2,split,split_ratio')
    args=parser.parse_args()
    from learning.dataset import DatasetStore
    from learning.line_model import split_rows
    store=DatasetStore();count=0
    with args.csv_path.open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            path=(args.csv_path.resolve().parent/row['image']).resolve()
            with Image.open(path) as source:image=source.convert('RGB')
            typ='double_row' if row.get('row2','').strip() else 'single_row'
            ratio=float(row['split_ratio']) if row.get('split_ratio','').strip() else split_rows(image,typ)[1]
            group=row.get('source_group','').strip() or hashlib.sha256(path.read_bytes()).hexdigest()
            store.save(image,row['row1'],row.get('row2',''),plate_type=typ,split=row.get('split','train'),
                       split_ratio=ratio,source_group=group,provenance='Human-labelled CSV: '+args.csv_path.name)
            count+=1
    print('Imported',count,'labelled plate images')

if __name__=='__main__':main()
