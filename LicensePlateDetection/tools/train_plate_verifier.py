"""Rebuild the plate object verifier from reviewed, explicitly labelled crops."""
from pathlib import Path
import json,sys
import cv2
import numpy as np
base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base))
from image_processing.object_verifier import descriptor

def main():
    data=base/'data/plate_verifier'
    records=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    rng=np.random.default_rng(20261105)
    features=[];labels=[];held=[]
    for record in records:
        path=(data/record['file']).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError('Dataset path outside the reference directory')
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Cannot load {path}')
        if record['split']=='heldout_negative':
            held.append((descriptor(image),record['label'],record['file']))
            continue
        for aug in range(18):
            a=image
            if aug:
                a=np.clip(a.astype(float)*rng.uniform(.7,1.3)+rng.uniform(-20,20),0,255).astype(np.uint8)
                if aug%3==0:a=cv2.GaussianBlur(a,(3,3),rng.uniform(.4,1.0))
                if aug%4==0:a=cv2.copyMakeBorder(a,3,3,3,3,cv2.BORDER_REPLICATE)
            features.append(descriptor(a));labels.append(record['label'])
    X=np.array(features,np.float32);y=np.array(labels,np.float32)
    weights=np.linalg.solve(X.T@X+np.eye(X.shape[1],dtype=np.float32)*.05,X.T@y)
    np.savez(data/'object_model.npz',weights=weights)
    stats={'training_examples':len(y),'training_errors':int(np.sum(np.where(X@weights>0,1,-1)!=y)),
           'negative_holdout':[{'file':path,'margin':float(feature@weights)} for feature,label,path in held],
           'note':'Calibration examples and related augmentations are not independent test images'}
    (data/'metadata.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
