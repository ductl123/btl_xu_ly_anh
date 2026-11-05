"""Train all archive train images (including _aug) and hold out val/test."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,hashlib,json,math,random,time,os,sys
import cv2,numpy as np,torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:from learning.archive_detector import PlateDetector,SIZE,read_rgb,letterbox,yolo_to_xyxy,targets,loss_fn,decode,map_to_original,load_detector
except ModuleNotFoundError:from archive_detector import PlateDetector,SIZE,read_rgb,letterbox,yolo_to_xyxy,targets,loss_fn,decode,map_to_original,load_detector


def json_save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)


def iou(box,other):
    wh=np.maximum(0,np.minimum(box[2:4],other[:,2:4])-np.maximum(box[:2],other[:,:2]))
    inter=wh.prod(1);union=(box[2:4]-box[:2]).prod()+(other[:,2:4]-other[:,:2]).prod(1)-inter
    return inter/np.maximum(union,1e-8)


def metrics(predictions,truth,match_iou=.5,threshold=None):
    count=sum(len(b) for b in truth)
    ranked=sorted(((float(p[4]),i,p[:4]) for i,rows in enumerate(predictions) for p in rows),key=lambda a:-a[0])
    found=[np.zeros(len(b),bool) for b in truth];scores=[];correct=[]
    for score,i,box in ranked:
        candidates=iou(box,truth[i]);candidates[found[i]]=-1
        best=int(candidates.argmax()) if len(candidates) else -1
        hit=best>=0 and candidates[best]>=match_iou
        if hit:found[i][best]=True
        scores.append(score);correct.append(bool(hit))
    if not scores:return {'ap':0.,'precision':0.,'recall':0.,'f1':0.,'threshold':.2,'tp':0,'fp':0,'fn':count,'ground_truth':count}
    scores=np.asarray(scores);tp=np.cumsum(correct);precision=tp/np.arange(1,len(tp)+1);recall=tp/max(count,1)
    envelope=np.maximum.accumulate(precision[::-1])[::-1]
    ap=sum(envelope[np.searchsorted(recall,r)] if r<=recall[-1] else 0. for r in np.linspace(0,1,101))/101
    f1=2*precision*recall/np.maximum(precision+recall,1e-8)
    if threshold is None:index=int(np.argmax(f1));threshold=float(scores[index])
    else:index=int((scores>=threshold).sum())-1
    t=int(tp[index]) if index>=0 else 0;p=float(precision[index]) if index>=0 else 0.;r=float(recall[index]) if index>=0 else 0.
    return {'ap':float(ap),'precision':p,'recall':r,'f1':float(f1[index]) if index>=0 else 0.,
            'threshold':float(threshold),'tp':t,'fp':index+1-t,'fn':count-t,'ground_truth':count}


def prepare(records,out):
    images=np.lib.format.open_memmap(out/'images.npy',mode='w+',dtype=np.uint8,shape=(len(records),SIZE,SIZE,3))
    labels=np.lib.format.open_memmap(out/'targets.npy',mode='w+',dtype=np.float32,shape=(len(records),5,SIZE//8,SIZE//8))
    actual=[];geometry=[]
    for i,record in enumerate(records):
        p=Path(record['image']);assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
        image=read_rgb(p);h,w=image.shape[:2]
        boxes=yolo_to_xyxy(record['boxes'],w,h)
        canvas,geo=letterbox(image);sx,sy,x,y=geo
        images[i]=canvas;labels[i]=targets(boxes*np.array([sx,sy,sx,sy])+np.array([x,y,x,y]))
        actual.append(boxes);geometry.append((geo,w,h))
        if (i+1)%500==0:print('Cached',i+1,'/',len(records),flush=True)
    images.flush();labels.flush()
    return images,labels,actual,geometry


def batch_from(images,indices):
    return torch.from_numpy(np.asarray(images[indices])).permute(0,3,1,2).float().div_(255.)


@torch.inference_mode()
def infer(model,images,indices,geometry,batch_size):
    model.eval();results=[]
    for start in range(0,len(indices),batch_size):
        idx=indices[start:start+batch_size]
        pred=decode(model(batch_from(images,idx)),score=.025,top_k=80)
        results.extend(map_to_original(item,*geometry[int(i)]) for item,i in zip(pred,idx))
    return results


def run(manifest,out,epochs,batch_size):
    records=json.loads(Path(manifest).read_text(encoding='utf-8'));out=Path(out);out.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(set);hashes=defaultdict(set)
    for row in records:
        assert 'boxes' in row
        groups[row['source_group']].add(row['split']);hashes[row['sha256']].add(row['split'])
    assert all(len(v)==1 for v in list(groups.values())+list(hashes.values()))
    split={name:np.array([i for i,r in enumerate(records) if r['split']==name],dtype=np.int32) for name in ['train','val','test']}
    assert all(len(x) for x in split.values())
    assert not any(r['augmented'] and r['split']!='train' for r in records)
    seed=20261105;random.seed(seed);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    images,labels,truth,geometry=prepare(records,out)
    model=PlateDetector().to(memory_format=torch.channels_last)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00015)
    history=[];best=-1.;best_epoch=0;start=time.monotonic()
    details={'architecture':'archive_plate_detector_v1','image_size':SIZE,'stride':8,'seed':seed,
        'image_counts':{k:len(v) for k,v in split.items()},'augmented_train':sum(r['augmented'] for r in records),
        'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),'epochs_requested':epochs,'device':'cpu'}
    val_truth=[truth[i] for i in split['val']]
    for epoch in range(1,epochs+1):
        model.train();losses=[];order=rng.permutation(split['train'])
        for offset in range(0,len(order),batch_size):
            idx=order[offset:offset+batch_size];x=batch_from(images,idx);y=torch.from_numpy(np.asarray(labels[idx]))
            optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(x),y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite detector loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        predictions=infer(model,images,split['val'],geometry,batch_size)
        val=metrics(predictions,val_truth);ap75=metrics(predictions,val_truth,.75)['ap']
        row={'epoch':epoch,'loss':float(np.mean(losses)),'val_ap50':val['ap'],'val_ap75':ap75,
             'val_precision':val['precision'],'val_recall':val['recall'],'threshold':val['threshold'],
             'elapsed_seconds':round(time.monotonic()-start,1)}
        history.append(row);json_save(out/'history.json',history)
        print(json.dumps(row),flush=True)
        selection=val['ap']+.1*ap75
        if selection>best:
            best=selection;best_epoch=epoch;val_best=val
            torch.save({**details,'epoch':epoch,'threshold':val['threshold'],'validation':val,
                        'state_dict':model.state_dict()},out/'best.pth')
    model,checkpoint=load_detector(out/'best.pth')
    predictions=infer(model,images,split['test'],geometry,batch_size)
    test_truth=[truth[i] for i in split['test']];threshold=checkpoint['threshold']
    test=metrics(predictions,test_truth,threshold=threshold)
    test['ap75']=metrics(predictions,test_truth,.75,threshold)['ap']
    test['map_50_95']=float(np.mean([metrics(predictions,test_truth,float(i),threshold)['ap'] for i in np.arange(.5,.951,.05)]))
    pred_rows=[{'image':records[int(i)]['image'],'truth_xyxy':truth[int(i)].tolist(),'detections_xyxy_score':d.tolist()}
               for i,d in zip(split['test'],predictions)]
    json_save(out/'test_predictions.json',pred_rows)
    report={**details,'epochs_completed':len(history),'best_epoch':best_epoch,'validation':val_best,'test':test,
            'test_used_for_selection':False,'elapsed_seconds':time.monotonic()-start,
            'best_sha256':hashlib.sha256((out/'best.pth').read_bytes()).hexdigest()}
    json_save(out/'report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--epochs',default=16,type=int)
    p.add_argument('--batch-size',default=16,type=int);a=p.parse_args();run(a.manifest,a.output,a.epochs,a.batch_size)
