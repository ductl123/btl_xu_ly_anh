"""OCR drafts for annotation only: this module never saves a label."""
def suggest_rows(image,plate_type,split_ratio=None):
    from .confirmed_crop import confirmed_crop_rows
    confirmed = confirmed_crop_rows(image, plate_type)
    if confirmed:
        return {'row1': confirmed[0], 'row2': confirmed[1], 'scores': [],
                'model_version': 'confirmed-crop', 'status': 'confirmed_crop'}
    from .line_model import engine,split_rows,predict_lines,_LOCK
    model,charset,device,version=engine()
    rows,_=split_rows(image,plate_type,split_ratio)
    with _LOCK:
        result=predict_lines(model,charset,device,[row[0] for row in rows],[False,True] if len(rows)==2 else [False])
    return {'row1':result[0]['text'],'row2':result[1]['text'] if len(result)==2 else '',
            'scores':[r['score'] for r in result],'model_version':version,'status':'unconfirmed_draft'}
