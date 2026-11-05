"""Explicitly labelled plate images with vehicle/photo-separated data splits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager

from PIL import Image

ALPHABET='0123456789ABCDEFGHKLMNPSTUVXYZ'
BASE=Path(__file__).resolve().parents[1]
SPLITS=('train','validation','test')


def clean_label(text):
    text=re.sub(r'[\s.\-]','',text.upper())
    if not text or any(c not in ALPHABET for c in text):
        raise ValueError('Nhãn chỉ được chứa các chữ và số trong bộ 30 ký tự của dự án.')
    if len(text)>12:
        raise ValueError('Một dòng nhãn quá dài cho biển số; hãy kiểm tra ảnh và cách tách dòng.')
    return text


def image_digest(image):
    image=image.convert('RGB')
    return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()


class DatasetStore:
    def __init__(self,root=None):
        self.root=Path(root) if root is not None else BASE/'data/ocr_training'
        self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'images').mkdir(exist_ok=True)
        self.db=self.root/'labels.sqlite3'
        with self.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS examples (
                id TEXT PRIMARY KEY, image_path TEXT NOT NULL, image_sha TEXT NOT NULL,
                source_group TEXT NOT NULL, plate_id TEXT NOT NULL, row1 TEXT NOT NULL,
                row2 TEXT NOT NULL, plate_type TEXT NOT NULL, split_ratio REAL,
                split TEXT NOT NULL, created_at TEXT NOT NULL, provenance TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1)''')

    @contextmanager
    def connect(self):
        conn=sqlite3.connect(self.db,timeout=30)
        conn.row_factory=sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def list_examples(self,split=None):
        query='SELECT * FROM examples WHERE active=1'
        args=[]
        if split is not None:
            if split not in SPLITS:raise ValueError('Unknown dataset split')
            query+=' AND split=?';args.append(split)
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query+' ORDER BY created_at,id',args)]

    def save(self,image,row1,row2='',plate_type='double_row',split='train',split_ratio=.5,
             source_group=None,provenance='Người dùng xác nhận thủ công'):
        if split not in SPLITS:raise ValueError('Tập dữ liệu không hợp lệ')
        if plate_type not in ('single_row','double_row'):raise ValueError('Loại biển không hợp lệ')
        row1=clean_label(row1)
        if plate_type=='double_row':
            row2=clean_label(row2)
            if not .25<=float(split_ratio)<=.75:raise ValueError('Vị trí tách dòng không hợp lệ')
            if not (2<=len(row1)<=6 and 3<=len(row2)<=6):raise ValueError('Hãy kiểm tra số ký tự của hai dòng')
        else:
            row2='';split_ratio=None
            if not 6<=len(row1)<=12:raise ValueError('Biển một dòng cần từ 6 đến 12 ký tự')
        image=image.convert('RGB')
        if min(image.size)<20:raise ValueError('Ảnh biển quá nhỏ để lưu huấn luyện')
        sha=image_digest(image);plate_id=row1+row2;source_group=source_group or sha
        ident=hashlib.sha256((sha+'|'+row1+'|'+row2).encode()).hexdigest()[:24]
        relative='images/'+sha+'.png'
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            related=conn.execute('''SELECT split FROM examples WHERE active=1
                AND (plate_id=? OR source_group=? OR image_sha=?)''',(plate_id,source_group,sha)).fetchall()
            if any(row['split']!=split for row in related):
                raise ValueError('Cùng một biển hoặc cùng ảnh gốc đã nằm ở tập khác. Giữ chúng trong cùng tập để tránh trùng dữ liệu train/kiểm tra.')
            previous=conn.execute('SELECT id FROM examples WHERE active=1 AND image_sha=?',(sha,)).fetchall()
            if any(row['id']!=ident for row in previous):
                raise ValueError('Ảnh này đã có nhãn khác. Xóa nhãn cũ trong danh sách trước khi sửa.')
            path=self.root/relative
            if not path.exists():image.save(path)
            conn.execute('''INSERT OR REPLACE INTO examples
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)''',
                (ident,relative,sha,source_group,plate_id,row1,row2,plate_type,split_ratio,
                 split,datetime.now(timezone.utc).isoformat(),provenance))
        return ident

    def remove(self,ident):
        with self.connect() as conn:
            conn.execute('UPDATE examples SET active=0 WHERE id=?',(ident,))

    def image_path(self,record):
        path=(self.root/record['image_path']).resolve()
        if not path.is_relative_to(self.root.resolve()):raise ValueError('Invalid dataset image path')
        return path

    def load_lines(self,record):
        from .line_model import split_rows
        with Image.open(self.image_path(record)) as source:image=source.convert('RGB')
        if image_digest(image)!=record['image_sha']:raise ValueError('Ảnh đã thay đổi sau khi gán nhãn')
        rows,_=split_rows(image,record['plate_type'],record['split_ratio'])
        texts=[record['row1']]+([record['row2']] if record['row2'] else [])
        return [(row[0],text) for row,text in zip(rows,texts)]

    def export_manifest(self,path):
        Path(path).write_text(json.dumps(self.list_examples(),ensure_ascii=False,indent=2),encoding='utf-8')


def verify_splits(records):
    for field in ('plate_id','source_group','image_sha'):
        groups={}
        for record in records:
            value=record[field];split=record['split']
            if value in groups and groups[value]!=split:
                raise ValueError('Dữ liệu bị trùng giữa các tập theo '+field)
            groups[value]=split


def manifest_digest(records):
    fields=('id','image_sha','plate_id','row1','row2','split','split_ratio')
    payload=[{k:r[k] for k in fields} for r in sorted(records,key=lambda x:x['id'])]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
