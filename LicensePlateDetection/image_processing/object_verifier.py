"""Spatial gradient descriptor and a small regularized linear object verifier."""
import cv2
import numpy as np

def descriptor(gray):
    gray=cv2.resize(gray,(96,64),interpolation=cv2.INTER_AREA)
    gray=cv2.createCLAHE(clipLimit=2.,tileGridSize=(4,4)).apply(gray)
    gx=cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,1,0)
    gy=cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,0,1)
    magnitude=np.hypot(gx,gy)
    angle=(np.arctan2(gy,gx)%np.pi)*9/np.pi
    bins=np.floor(angle).astype(int)%9;fraction=angle-np.floor(angle)
    cell=np.repeat(np.arange(8),8)[:,None]*12+np.repeat(np.arange(12),8)[None,:]
    histogram=(np.bincount((cell*9+bins).ravel(),(magnitude*(1-fraction)).ravel(),minlength=96*9)
               +np.bincount((cell*9+(bins+1)%9).ravel(),(magnitude*fraction).ravel(),minlength=96*9)).reshape(96,9)
    histogram/=np.sqrt((histogram**2).sum(axis=1,keepdims=True)+1e-6)
    vector=histogram.ravel().astype(np.float32)
    vector/=np.linalg.norm(vector)+1e-6
    return np.r_[vector,1.].astype(np.float32)


from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def _weights():
    path=Path(__file__).resolve().parents[1]/'data/plate_verifier/object_model.npz'
    if not path.is_file():
        return None
    with np.load(path,allow_pickle=False) as data:
        weights=data['weights'].copy()
    if weights.shape!=(865,) or not np.isfinite(weights).all():
        raise ValueError('Invalid plate verifier model')
    return weights


def object_margin(gray):
    """Signed evidence from labelled object patches, not an OCR confidence."""
    weights=_weights()
    return None if weights is None else float(descriptor(gray)@weights)
