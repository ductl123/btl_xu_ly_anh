"""Fine-tune on all labelled training images; never train on validation/test."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
import random
import threading
import uuid
from collections import Counter

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from filelock import FileLock

from .dataset import DatasetStore, verify_splits, manifest_digest
from .line_model import ALPHABET, MODEL_DIR, load_model, prepare_batch, canonical_log_probs, predict_lines


def atomic_json(path,payload):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(payload,handle,ensure_ascii=False,indent=2);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def save_weights(path,state,metadata):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        torch.save({'state_dict':state,'metadata':metadata},handle)
        handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        next_row=[i]
        for j,cb in enumerate(b,1):
            next_row.append(min(next_row[-1]+1,row[j]+1,row[j-1]+(ca!=cb)))
        row=next_row
    return row[-1]


def evaluate(model,charset,device,store,records):
    model.eval();errors=chars=correct=0;details=[]
    for record in records:
        lines=store.load_lines(record)
        predictions=predict_lines(model,charset,device,[line[0] for line in lines],
                                  [False,True] if len(lines)==2 else [False])
        expected=''.join(line[1] for line in lines)
        observed=''.join(p['text'] for p in predictions)
        errors+=edit_distance(expected,observed);chars+=len(expected);correct+=expected==observed
        details.append({'id':record['id'],'expected':expected,'observed':observed})
    return {'plates':len(records),'correct':correct,'exact_accuracy':correct/max(1,len(records)),
            'character_error_rate':errors/max(1,chars),'character_errors':errors,
            'characters':chars,'details':details}


def evaluate_synthetic(model,charset,device,examples):
    model.eval();errors=characters=correct=0
    for start in range(0,len(examples),8):
        batch=examples[start:start+8]
        predictions=predict_lines(model,charset,device,[x[0] for x in batch],[x[2]=='serial' for x in batch])
        for (_,label,_),prediction in zip(batch,predictions):
            errors+=edit_distance(label,prediction['text']);characters+=len(label)
            correct+=label==prediction['text']
    return {'lines':len(examples),'correct':correct,'characters':characters,
            'character_errors':errors,'character_error_rate':errors/max(1,characters)}


def augment(image,rng):
    image=ImageEnhance.Brightness(image).enhance(rng.uniform(.8,1.2))
    image=ImageEnhance.Contrast(image).enhance(rng.uniform(.85,1.2))
    if rng.random()<.3:image=image.filter(ImageFilter.GaussianBlur(rng.uniform(.1,.6)))
    if rng.random()<.4:
        background=tuple(np.percentile(np.asarray(image),75,axis=(0,1)).astype(int))
        image=image.rotate(rng.uniform(-2,2),Image.Resampling.BICUBIC,fillcolor=background)
    return image


def _rank(validation,training,synthetic=None):
    # Validation comes first. On an exact validation tie, require less training
    # error before preferring a new domain-adapted model to the starting model.
    return (round(validation['character_error_rate'],8),-validation['correct'],
            round(training['character_error_rate'],8),
            round(synthetic['character_error_rate'],8) if synthetic else 0.)


def train(store=None,model_dir=MODEL_DIR,epochs=20,batch_size=8,learning_rate=.0001,
          device='auto',cancel=None,progress=None,seed=20261105,
          use_synthetic=False,synthetic_root=None,synthetic_per_epoch=256,start_checkpoint=None):
    if not 1<=epochs<=500 or not 1<=batch_size<=64 or not 1e-7<=learning_rate<=.01:
        raise ValueError('Thông số huấn luyện nằm ngoài giới hạn cho phép')
    if not 0<=synthetic_per_epoch<=100000:raise ValueError('Giới hạn ảnh giả mỗi vòng phải từ 0 đến 100.000')
    store=store or DatasetStore();model_dir=Path(model_dir);model_dir.mkdir(parents=True,exist_ok=True)
    cancel=cancel or threading.Event();progress=progress or (lambda event:None)
    with FileLock(str(model_dir/'training.lock'),timeout=0):
        records=store.list_examples();verify_splits(records)
        training=[r for r in records if r['split']=='train']
        validation=[r for r in records if r['split']=='validation']
        if not training:raise ValueError('Chưa có ảnh gán nhãn thuộc tập Huấn luyện')
        if not validation:raise ValueError('Cần ít nhất một biển khác ở tập Kiểm định để so sánh mô hình mới và cũ')
        torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed)
        torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
        model,charset,selected_device,starting_version=load_model(model_dir,device)
        criterion=nn.CTCLoss(blank=0,zero_infinity=False)
        examples=[(image,text) for record in training for image,text in store.load_lines(record)]
        synthetic_store=None;synthetic_records=[];probe=[];synthetic_cursor=0;synthetic_order=[]
        used_synthetic=set();synthetic_counts=Counter()
        if use_synthetic:
            from .synthetic import SyntheticStore,DEFAULT_ROOT
            synthetic_store=SyntheticStore(synthetic_root or DEFAULT_ROOT)
            held_out={r['plate_id'] for r in records if r['split']!='train'}
            synthetic_records=synthetic_store.records(exclude_plate_ids=held_out)
            if not synthetic_records:raise ValueError('Chưa có lô ảnh giả hoàn tất; hãy sinh dữ liệu trước')
            synthetic_order=list(range(len(synthetic_records)));rng.shuffle(synthetic_order)
            probe_ids=random.Random(seed+1).sample(range(len(synthetic_records)),min(64,len(synthetic_records)))
            probe=[(*synthetic_store.load(synthetic_records[i]),synthetic_records[i]['kind']) for i in probe_ids]
        targets={c:i+1 for i,c in enumerate(ALPHABET)}
        baseline_val=evaluate(model,charset,selected_device,store,validation)
        baseline_train=evaluate(model,charset,selected_device,store,training)
        baseline_synthetic=evaluate_synthetic(model,charset,selected_device,probe) if probe else None
        baseline_version=starting_version
        best_rank=_rank(baseline_val,baseline_train,baseline_synthetic);best_state=None;best_epoch=0
        best_val=baseline_val;best_train=baseline_train;best_synthetic=baseline_synthetic;history=[]
        if start_checkpoint is not None:
            model,charset,selected_device,starting_version=load_model(model_dir,device,checkpoint=start_checkpoint)
        # Replay real data and adapt the reader while retaining generic CNN features.
        for parameter in model.FeatureExtraction.parameters():parameter.requires_grad=False
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate,weight_decay=.0001)
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        run=model_dir/'runs'/run_id;run.mkdir(parents=True)
        dataset_hash=manifest_digest(records)
        initial_head=model.Prediction.weight.detach().cpu().clone()
        info={'version':run_id,'starting_version':starting_version,'architecture':'VGG-BiLSTM-CTC',
              'dataset_hash':dataset_hash,'training_ids':[r['id'] for r in training],
              'validation_ids':[r['id'] for r in validation],
              'test_ids_excluded':[r['id'] for r in records if r['split']=='test'],
              'epochs_requested':epochs,'batch_size':batch_size,'learning_rate':learning_rate,
              'device':str(selected_device),'frozen_visual_backbone':True,'seed':seed}
        info['trained_character_counts']=dict(Counter(''.join(text for _,text in examples)))
        info.update({'baseline_version':baseline_version,'uses_synthetic':bool(synthetic_records),
                     'synthetic_available':len(synthetic_records),'synthetic_per_epoch':synthetic_per_epoch,
                     'synthetic_mix':'balanced real replay and synthetic lines',
                     'synthetic_digest':synthetic_store.digest() if synthetic_store else None})
        if synthetic_store:
            atomic_json(run/'synthetic_batches.json',[metadata for _,metadata in synthetic_store.batches()])

        atomic_json(run/'dataset_snapshot.json',records)
        progress({'stage':'baseline','validation':baseline_val,'training':baseline_train,'run_id':run_id})
        for epoch in range(1,epochs+1):
            if cancel.is_set():break
            model.train();model.FeatureExtraction.eval()
            real_order=list(range(len(examples)));rng.shuffle(real_order)
            selected=[]
            if synthetic_records:
                limit=min(synthetic_per_epoch or len(synthetic_records),len(synthetic_records))
                for _ in range(limit):
                    if synthetic_cursor==len(synthetic_order):
                        rng.shuffle(synthetic_order);synthetic_cursor=0
                    selected.append(synthetic_order[synthetic_cursor]);synthetic_cursor+=1
            jobs=[('real',real_order[i%len(real_order)]) for i in range(max(len(real_order),len(selected)))]
            jobs.extend(('synthetic',i) for i in selected);rng.shuffle(jobs)
            losses=[]
            for start in range(0,len(jobs),batch_size):
                if cancel.is_set():break
                batch=[]
                for kind,index in jobs[start:start+batch_size]:
                    if kind=='real':batch.append(examples[index])
                    else:
                        record=synthetic_records[index]
                        batch.append(synthetic_store.load(record))
                        used_synthetic.add(record['id']);synthetic_counts.update(record['text'])
                images=[augment(image,rng) for image,_ in batch]
                tensors,input_lengths,_=prepare_batch(images,selected_device)
                labels=[text for _,text in batch]
                encoded=torch.tensor([targets[c] for text in labels for c in text],dtype=torch.long,device=selected_device)
                lengths=torch.tensor([len(text) for text in labels],dtype=torch.long)
                if any(int(n)<len(text)+sum(a==b for a,b in zip(text,text[1:])) for n,text in zip(input_lengths,labels)):
                    raise ValueError('Một ảnh dòng quá hẹp so với nhãn; hãy kiểm tra lại cách cắt')
                optimizer.zero_grad(set_to_none=True)
                logs=canonical_log_probs(model(tensors),charset).transpose(0,1)
                loss=criterion(logs,encoded,input_lengths,lengths)
                if not torch.isfinite(loss):raise RuntimeError('CTC loss không hữu hạn; không lưu hoặc kích hoạt trọng số lỗi')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();losses.append(float(loss.detach().cpu()))
            if cancel.is_set():break
            val=evaluate(model,charset,selected_device,store,validation)
            tr=evaluate(model,charset,selected_device,store,training)
            syn=evaluate_synthetic(model,charset,selected_device,probe) if probe else None
            row={'epoch':epoch,'epochs':epochs,'loss':float(np.mean(losses)),
                 'validation':val,'training':tr,'synthetic_training':syn}
            history.append(row);rank=_rank(val,tr,syn)
            if rank<best_rank:
                best_rank=rank;best_epoch=epoch;best_val=val;best_train=tr;best_synthetic=syn
                best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            progress({'stage':'epoch',**row})
            atomic_json(run/'history.json',history)
        changed=float(torch.linalg.vector_norm(model.Prediction.weight.detach().cpu()-initial_head))
        metadata={**info,'epochs_completed':len(history),'best_epoch':best_epoch,'head_weight_change_l2':changed,
                  'cancelled':cancel.is_set(),'synthetic_unique_used':len(used_synthetic),
                  'synthetic_character_counts':dict(synthetic_counts)}
        if synthetic_store:atomic_json(run/'synthetic_ids_used.json',sorted(used_synthetic))
        last_state={key:value.detach().cpu() for key,value in model.state_dict().items()}
        save_weights(run/'last.pth',last_state,metadata)
        activated=False
        if best_state is not None and not cancel.is_set():
            save_weights(run/'best.pth',best_state,metadata)
            checkpoint=run/'best.pth'
            pointer={'checkpoint':str(checkpoint.relative_to(model_dir)).replace('\\','/'),'version':run_id,
                     'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'dataset_hash':dataset_hash,
                     'validation':best_val,'training':best_train}
            pointer['trained_character_counts']=info['trained_character_counts']
            atomic_json(model_dir/'active.json',pointer);activated=True
        report={**metadata,'activated':activated,'baseline_validation':baseline_val,
                'baseline_training':baseline_train,'best_validation':best_val,'best_training':best_train,
                'baseline_synthetic_training':baseline_synthetic,'best_synthetic_training':best_synthetic,
                'run_dir':str(run),'last_checkpoint':str(run/'last.pth')}
        atomic_json(run/'report.json',report)
        atomic_json(model_dir/'last_training_report.json',report)
        progress({'stage':'complete','report':report})
        return report
