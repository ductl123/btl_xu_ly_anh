"""Tkinter annotation/training workflow. Network output is never a ground truth."""
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
import threading
import time
import os
import tkinter as tk
from tkinter import ttk,filedialog,messagebox

from PIL import Image,ImageTk
from .dataset import DatasetStore

SPLIT_NAMES={'Huấn luyện':'train','Kiểm định':'validation','Kiểm tra cuối':'test'}


class TrainingWindow(tk.Toplevel):
    def __init__(self,app):
        super().__init__(app.root)
        self.app=app;self.store=DatasetStore();self.image=None;self.source_group=None
        self.training=False;self.cancel=threading.Event();self.preview_refs=[];self.pending_paths=[]
        self._suggestion_revision=0;self._suggestion_after=None;self._suggestion_closed=False
        self._preview_context=None;self.last_suggestion=None
        self.title('Gán nhãn ảnh và huấn luyện OCR');self.geometry(f'1100x{min(850,max(620,self.winfo_screenheight()-100))}');self.minsize(900,600)
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.kind=tk.StringVar(value='2 dòng');self.split=tk.DoubleVar(value=.5)
        self.partition=tk.StringVar(value='Huấn luyện')
        self.row1=tk.StringVar();self.row2=tk.StringVar();self.epochs=tk.StringVar(value='20')
        self.message=tk.StringVar(value='Chọn một vùng biển, xem ảnh rồi nhập nhãn đúng.')
        self.suggestion_message=tk.StringVar(value='Gợi ý OCR chưa được lưu; hãy nhìn ảnh, sửa rồi xác nhận.')
        self.synthetic_count=tk.StringVar(value='20000');self.synthetic_limit=tk.StringVar(value='256')
        from .synthetic import SyntheticStore
        self.use_synthetic=tk.BooleanVar(value=SyntheticStore().count()>0)
        self.synthetic_info=tk.StringVar()
        self.stats=tk.StringVar();self.progress_text=tk.StringVar(value='Chưa chạy huấn luyện')
        viewport=ttk.Frame(self);viewport.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(viewport,highlightthickness=0)
        scroll=ttk.Scrollbar(viewport,orient='vertical',command=self.canvas.yview)
        scroll.pack(side='right',fill='y');self.canvas.pack(side='left',fill='both',expand=True)
        self.canvas.configure(yscrollcommand=scroll.set)
        self.body=ttk.Frame(self.canvas)
        self.body_window=self.canvas.create_window((0,0),window=self.body,anchor='nw')
        self.body.bind('<Configure>',lambda _:self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>',lambda e:self.canvas.itemconfigure(self.body_window,width=e.width))
        self.bind('<MouseWheel>',lambda e:self.canvas.yview_scroll(-int(e.delta/120),'units') if not isinstance(e.widget,ttk.Treeview) else None)
        top=ttk.Frame(self.body,padding=12);top.pack(fill='x')
        ttk.Label(top,text='Ảnh có nhãn → huấn luyện bổ sung → lưu trọng số',font=('Segoe UI',13,'bold')).pack(anchor='w')
        ttk.Label(top,text='Huấn luyện để học; Kiểm định để chọn phiên bản; Kiểm tra cuối không tham gia học hoặc chọn mô hình.').pack(anchor='w')
        ttk.Label(top,textvariable=self.stats).pack(anchor='w',pady=4)
        actions=ttk.Frame(top);actions.pack(fill='x')
        self.current_button=ttk.Button(actions,text='Lấy biển đang chọn ở màn hình chính',command=self.take_current);self.current_button.pack(side='left',padx=3)
        self.open_button=ttk.Button(actions,text='Mở nhiều ảnh biển đã cắt',command=self.open_image);self.open_button.pack(side='left',padx=3)
        ttk.Label(actions,text='Loại biển:').pack(side='left',padx=(12,3))
        self.type_combo=ttk.Combobox(actions,textvariable=self.kind,values=['1 dòng','2 dòng'],state='readonly',width=8)
        self.type_combo.pack(side='left');self.type_combo.bind('<<ComboboxSelected>>',lambda _:self.preview_and_suggest())
        self.rotate_button=ttk.Button(actions,text='Xoay 90°',command=self.rotate_image)
        self.rotate_button.pack(side='left',padx=8)
        self.suggest_button=ttk.Button(actions,text='Gợi ý OCR',command=self.schedule_suggestion)
        self.suggest_button.pack(side='left',padx=4)
        previews=ttk.Frame(self.body,padding=8);previews.pack(fill='x')
        self.plate_preview=ttk.Label(previews,text='Ảnh biển');self.plate_preview.pack(side='left',padx=12)
        lines=ttk.Frame(previews);lines.pack(side='left',fill='x',expand=True)
        self.first_preview=ttk.Label(lines,text='Dòng trên');self.first_preview.pack(anchor='w')
        self.second_preview=ttk.Label(lines,text='Dòng dưới');self.second_preview.pack(anchor='w')
        self.slider=ttk.Scale(self.body,from_=.25,to=.75,variable=self.split,command=lambda _:self.preview())
        self.slider.pack(fill='x',padx=25)
        self.slider.bind('<ButtonRelease-1>',lambda _:self.schedule_suggestion())
        ttk.Label(self.body,text='Điều chỉnh thanh tách sao cho mỗi ảnh nhỏ chứa trọn một dòng chữ.').pack(anchor='w',padx=20)
        form=ttk.Frame(self.body,padding=12);form.pack(fill='x')
        ttk.Label(form,text='Nhãn dòng trên / một dòng:').grid(row=0,column=0,sticky='w')
        self.first_entry=ttk.Entry(form,textvariable=self.row1,width=22,font=('Consolas',14));self.first_entry.grid(row=0,column=1,padx=8)
        ttk.Label(form,text='Nhãn dòng dưới:').grid(row=1,column=0,sticky='w')
        self.second_entry=ttk.Entry(form,textvariable=self.row2,width=22,font=('Consolas',14));self.second_entry.grid(row=1,column=1,padx=8,pady=5)
        ttk.Label(form,text='Lưu vào tập:').grid(row=0,column=2,padx=(20,5))
        self.split_combo=ttk.Combobox(form,textvariable=self.partition,values=list(SPLIT_NAMES),width=16,state='readonly');self.split_combo.grid(row=0,column=3)
        self.save_button=ttk.Button(form,text='Lưu nhãn đã xác nhận',command=self.save);self.save_button.grid(row=1,column=3,padx=5)
        ttk.Label(self.body,textvariable=self.message,wraplength=1000,foreground='#285480').pack(fill='x',padx=15)
        ttk.Label(self.body,textvariable=self.suggestion_message,wraplength=1000,foreground='#926000').pack(fill='x',padx=15)
        table=ttk.Frame(self.body,padding=10);table.pack(fill='both',expand=True)
        self.tree=ttk.Treeview(table,columns=('label','split','time'),show='headings',height=5)
        for col,title,width in [('label','Biển số đúng',220),('split','Tập dữ liệu',140),('time','Ngày lưu',230)]:
            self.tree.heading(col,text=title);self.tree.column(col,width=width)
        self.tree.pack(side='left',fill='both',expand=True)
        scroll=ttk.Scrollbar(table,orient='vertical',command=self.tree.yview);scroll.pack(side='right',fill='y');self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind('<<TreeviewSelect>>',self.select_record)
        synthetic=ttk.Frame(self.body,padding=(12,4));synthetic.pack(fill='x')
        ttk.Label(synthetic,text='Ảnh giả cần sinh:').pack(side='left')
        self.synthetic_entry=ttk.Entry(synthetic,textvariable=self.synthetic_count,width=7);self.synthetic_entry.pack(side='left',padx=4)
        self.generate_button=ttk.Button(synthetic,text='Sinh ảnh giả',command=self.generate_synthetic);self.generate_button.pack(side='left',padx=5)
        self.preview_synthetic_button=ttk.Button(synthetic,text='Xem ảnh mẫu',command=self.open_synthetic_preview);self.preview_synthetic_button.pack(side='left',padx=5)
        self.synthetic_check=ttk.Checkbutton(synthetic,text='Trộn ảnh giả khi train',variable=self.use_synthetic);self.synthetic_check.pack(side='left',padx=8)
        ttk.Label(synthetic,text='Tối đa/vòng (0=tất cả):').pack(side='left')
        self.synthetic_limit_entry=ttk.Entry(synthetic,textvariable=self.synthetic_limit,width=6);self.synthetic_limit_entry.pack(side='left',padx=4)
        ttk.Label(self.body,textvariable=self.synthetic_info).pack(anchor='w',padx=15)
        bottom=ttk.Frame(self.body,padding=12);bottom.pack(fill='x')
        self.remove_button=ttk.Button(bottom,text='Bỏ mẫu đã chọn',command=self.remove);self.remove_button.pack(side='left')
        ttk.Label(bottom,text='Số vòng học:').pack(side='left',padx=(20,4))
        self.epoch_entry=ttk.Entry(bottom,textvariable=self.epochs,width=5);self.epoch_entry.pack(side='left')
        self.train_button=ttk.Button(bottom,text='Huấn luyện bổ sung',command=self.start_training);self.train_button.pack(side='left',padx=10)
        self.stop_button=ttk.Button(bottom,text='Dừng sau lượt hiện tại',command=self.cancel.set,state='disabled');self.stop_button.pack(side='left')
        self.test_button=ttk.Button(bottom,text='Đánh giá tập kiểm tra cuối',command=self.test);self.test_button.pack(side='right')
        self.bar=ttk.Progressbar(self.body,maximum=100);self.bar.pack(fill='x',padx=15)
        ttk.Label(self.body,textvariable=self.progress_text,wraplength=1000).pack(fill='x',padx=15,pady=8)
        self.refresh();self.take_current(quiet=True)

    def refresh(self):
        self.records={r['id']:r for r in self.store.list_examples()}
        for item in self.tree.get_children():self.tree.delete(item)
        reverse={v:k for k,v in SPLIT_NAMES.items()}
        for record in self.records.values():
            local=(datetime.fromisoformat(record['created_at'])+timedelta(hours=7)).strftime('%d/%m/%Y %H:%M')
            self.tree.insert('',tk.END,iid=record['id'],values=(record['plate_id'],reverse[record['split']],local))
        counts={s:sum(r['split']==s for r in self.records.values()) for s in SPLIT_NAMES.values()}
        from .synthetic import SyntheticStore
        self.synthetic_info.set(f'Ảnh giả hoàn tất: {SyntheticStore().count():,}. Chỉ dùng để học; kiểm định/kiểm tra vẫn là ảnh thật.')
        self.stats.set(f"Ảnh đã gán nhãn: {sum(counts.values())} | Huấn luyện {counts['train']} | Kiểm định {counts['validation']} | Kiểm tra cuối {counts['test']}")

    def load(self,path,source_group=None,plate_type='double_row',suggest=True):
        with Image.open(path) as image:self.image=image.convert('RGB')
        self.source_group=source_group or hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.kind.set('1 dòng' if plate_type=='single_row' else '2 dòng')
        self.row1.set('');self.row2.set('')
        from .line_model import split_rows
        _,ratio=split_rows(self.image,plate_type);self.split.set(ratio or .5)
        self.preview();self.message.set('Nhập nhãn đúng bằng cách nhìn ảnh; dấu chấm, gạch ngang sẽ được bỏ khi lưu.')
        if suggest:self.schedule_suggestion()

    def take_current(self,quiet=False):
        if self.training:return
        selected=self.app.tree.selection()
        idx=int(self.app.tree.item(selected[0])['values'][0]) if selected else None
        record=next((r for r in self.app.current_candidates if r['plate_index']==idx),None)
        if record and Path(record.get('cropped_image_path','')).is_file():
            source=Path(self.app.current_image_path)
            self.load(record['cropped_image_path'],hashlib.sha256(source.read_bytes()).hexdigest(),record['plate_type'])
        elif not quiet:self.message.set('Hãy chọn một dòng biển trong bảng kết quả, hoặc mở một ảnh biển đã cắt.')

    def open_image(self):
        if self.training:return
        paths=filedialog.askopenfilenames(parent=self,title='Chọn các ảnh chỉ chứa biển số',filetypes=[('Ảnh','*.png *.jpg *.jpeg *.webp *.bmp')])
        if paths:
            self.pending_paths=list(paths[1:])
            try:self.load(paths[0])
            except Exception as exc:messagebox.showerror('Không mở được ảnh',str(exc),parent=self)

    def preview(self):
        if self.image is None:return
        from .line_model import split_rows
        context=(id(self.image),self.kind.get(),round(float(self.split.get()),5))
        if context!=self._preview_context:
            previous=self.last_suggestion
            if previous:
                if self.row1.get()==previous['row1']:self.row1.set('')
                if self.row2.get()==previous['row2']:self.row2.set('')
            self.last_suggestion=None;self._preview_context=context;self._suggestion_revision+=1
            self.suggestion_message.set('Dòng ảnh đã thay đổi; gợi ý OCR chỉ là bản nháp chưa lưu.')
        typ='single_row' if self.kind.get()=='1 dòng' else 'double_row'
        rows,_=split_rows(self.image,typ,float(self.split.get()))
        self.preview_refs=[]
        for image,label,limit in [(self.image,self.plate_preview,(300,130)),(rows[0][0],self.first_preview,(430,55))]:
            copy=image.copy();copy.thumbnail(limit);ref=ImageTk.PhotoImage(copy)
            self.preview_refs.append(ref);label.configure(image=ref,text='')
        if len(rows)==2:
            copy=rows[1][0].copy();copy.thumbnail((430,55));ref=ImageTk.PhotoImage(copy)
            self.preview_refs.append(ref);self.second_preview.configure(image=ref,text='')
        else:self.second_preview.configure(image='',text='Biển một dòng')
        self.second_entry.configure(state='normal' if len(rows)==2 else 'disabled')

    def rotate_image(self):
        if self.training or self.image is None:return
        self.image=self.image.rotate(90,expand=True)
        self.row1.set('');self.row2.set('');self.split.set(.5);self.preview()
        self.message.set('Đã xoay ảnh. Xoay tiếp nếu cần rồi nhập nhãn khi chữ đã đúng chiều đọc.')
        self.schedule_suggestion()

    def save(self):
        if self.training:return
        if self.image is None:self.message.set('Chưa chọn ảnh biển');return
        try:
            self.store.save(self.image,self.row1.get(),self.row2.get(),
                plate_type='single_row' if self.kind.get()=='1 dòng' else 'double_row',
                split=SPLIT_NAMES[self.partition.get()],split_ratio=float(self.split.get()),source_group=self.source_group,
                provenance=json.dumps({'type':'user_confirmed','ocr_draft':self.last_suggestion,
                                       'confirmation':'Explicit Save button; OCR draft was not saved automatically'},ensure_ascii=False))
            self.refresh();self.message.set('Đã lưu nhãn. Ảnh mới sẽ tham gia lần huấn luyện tiếp theo nếu thuộc tập Huấn luyện.')
            if self.pending_paths:
                self.load(self.pending_paths.pop(0))
                self.message.set(f'Đã lưu ảnh trước. Nhập nhãn ảnh tiếp theo; còn {len(self.pending_paths)} ảnh trong hàng đợi.')
        except Exception as exc:messagebox.showerror('Chưa lưu nhãn',str(exc),parent=self)

    def select_record(self,_=None):
        if self.training:return
        selected=self.tree.selection()
        if not selected:return
        record=self.records[selected[0]]
        self.load(self.store.image_path(record),record['source_group'],record['plate_type'],suggest=False)
        self.row1.set(record['row1']);self.row2.set(record['row2']);self.split.set(record['split_ratio'] or .5)
        self.partition.set(next(k for k,v in SPLIT_NAMES.items() if v==record['split']));self.preview()

    def remove(self):
        if self.training:return
        for ident in self.tree.selection():self.store.remove(ident)
        self.refresh();self.message.set('Đã bỏ mẫu khỏi tập đang dùng. Ảnh được giữ lại, không xóa khỏi máy.')

    def busy(self,value):
        self.training=value;self.app.is_busy=value;self.app._set_buttons_state(not value)
        if value:
            self._suggestion_revision+=1
            if self._suggestion_after is not None:
                self.after_cancel(self._suggestion_after);self._suggestion_after=None
        for widget in [self.synthetic_entry,self.synthetic_limit_entry,self.synthetic_check]:
            widget.configure(state='disabled' if value else 'normal')
        for button in [self.current_button,self.open_button,self.rotate_button,self.suggest_button,self.generate_button,self.preview_synthetic_button,self.save_button,self.remove_button,self.train_button,self.test_button]:
            button.configure(state='disabled' if value else 'normal')
        self.stop_button.configure(state='normal' if value else 'disabled')

    def notify(self,event):
        if not self.winfo_exists():return
        if event['stage']=='synthetic_generate':
            self.bar['value']=100*event['generated']/event['total']
            self.progress_text.set(f"Đang sinh ảnh giả: {event['generated']:,}/{event['total']:,}")
        elif event['stage']=='baseline':self.progress_text.set('Đã đo mô hình hiện tại. Bắt đầu cập nhật trọng số…')
        elif event['stage']=='epoch':
            self.bar['value']=100*event['epoch']/event['epochs']
            val=event['validation']
            self.progress_text.set(f"Vòng {event['epoch']}/{event['epochs']} | Loss {event['loss']:.4f} | Kiểm định đúng {val['correct']}/{val['plates']} biển | Lỗi ký tự {val['character_error_rate']:.1%}")

    def start_training(self):
        if self.training or self.app.is_busy:return
        try:
            epochs=int(self.epochs.get());synthetic_limit=int(self.synthetic_limit.get())
        except ValueError:messagebox.showerror('Số vòng học','Nhập số nguyên từ 1 đến 500',parent=self);return
        synthetic_enabled=bool(self.use_synthetic.get())
        self.cancel.clear();self.busy(True);self.bar['value']=0
        self.progress_text.set('Đang nạp trọng số và kiểm tra tập dữ liệu…')
        def worker():
            try:
                from .trainer import train
                report=train(store=self.store,epochs=epochs,cancel=self.cancel,use_synthetic=synthetic_enabled,synthetic_per_epoch=synthetic_limit,
                             progress=lambda event:self.after(0,lambda e=event:self.notify(e)))
                self.after(0,lambda:self.finished(report))
            except Exception as exc:self.after(0,lambda msg=str(exc):self.failed(msg))
        threading.Thread(target=worker,daemon=True).start()

    def finished(self,report):
        self.busy(False);self.bar['value']=100 if not report['cancelled'] else self.bar['value']
        if report['cancelled']:message='Đã dừng. Mô hình đang dùng được giữ nguyên.'
        elif report['activated']:message='Đã lưu và dùng trọng số mới. Chọn lại ảnh để nhận dạng bằng phiên bản mới.'
        else:message='Đã lưu trọng số thử nghiệm. Chưa đủ điều kiện chọn phiên bản mới nên tiếp tục dùng mô hình trước.'
        self.progress_text.set(message+' Báo cáo: '+str(Path(report['run_dir'])/'report.json'))

    def failed(self,message):
        self.busy(False);self.progress_text.set('Chưa hoàn tất: '+message)
        messagebox.showerror('Huấn luyện / đánh giá',message,parent=self)

    def test(self):
        if self.training or self.app.is_busy:return
        self.busy(True);self.progress_text.set('Đang đánh giá tập kiểm tra cuối, không cập nhật trọng số…')
        def worker():
            try:
                from .line_model import load_model,MODEL_DIR
                from .trainer import evaluate,atomic_json
                records=self.store.list_examples('test')
                if not records:raise ValueError('Chưa có ảnh thuộc tập Kiểm tra cuối')
                model,chars,device,version=load_model()
                report={'version':version,**evaluate(model,chars,device,self.store,records)}
                atomic_json(MODEL_DIR/'last_test_evaluation.json',report)
                self.after(0,lambda:self.test_finished(report))
            except Exception as exc:self.after(0,lambda msg=str(exc):self.failed(msg))
        threading.Thread(target=worker,daemon=True).start()

    def test_finished(self,report):
        self.busy(False)
        self.progress_text.set(f"Kiểm tra cuối: đúng {report['correct']}/{report['plates']} biển; lỗi ký tự {report['character_error_rate']:.1%}. Trọng số không thay đổi.")

    def preview_and_suggest(self):
        self.preview();self.schedule_suggestion()

    def schedule_suggestion(self):
        if self.training or self.image is None or self._suggestion_closed:return
        if self._suggestion_after is not None:self.after_cancel(self._suggestion_after)
        self._suggestion_revision+=1
        self._suggestion_after=self.after(150,self._start_suggestion)

    def _start_suggestion(self):
        self._suggestion_after=None
        if self.training or self.image is None or self._suggestion_closed:return
        token=self._suggestion_revision;image=self.image.copy()
        kind='single_row' if self.kind.get()=='1 dòng' else 'double_row';ratio=float(self.split.get())
        initial=(self.row1.get(),self.row2.get())
        self.suggestion_message.set('Đang lấy gợi ý OCR; bạn vẫn có thể nhập nhãn đúng…')
        def worker():
            try:
                from .suggestions import suggest_rows
                result=suggest_rows(image,kind,ratio)
                if not self._suggestion_closed:
                    self.after(0,lambda:self._apply_suggestion(token,initial,result))
            except Exception as exc:
                if not self._suggestion_closed:
                    try:self.after(0,lambda msg=str(exc):self._suggestion_error(token,msg))
                    except (RuntimeError,tk.TclError):pass
        threading.Thread(target=worker,daemon=True).start()

    def _suggestion_error(self,token,message):
        if not self._suggestion_closed and not self.training and token==self._suggestion_revision:
            self.suggestion_message.set('Không lấy được gợi ý: '+message)

    def _apply_suggestion(self,token,initial,result):
        if self._suggestion_closed or self.training or token!=self._suggestion_revision:return
        self.last_suggestion=result
        # Never overwrite a human edit, or replace an already labelled sample.
        if not initial[0] and not self.row1.get():self.row1.set(result['row1'])
        if not initial[1] and not self.row2.get():self.row2.set(result['row2'])
        if result.get('status') == 'confirmed_crop':
            self.suggestion_message.set('Đã tìm thấy nhãn xác nhận của ảnh cắt trùng khớp.')
            return
        scores=' / '.join(f'{score:.2f}' for score in result['scores'])
        self.suggestion_message.set('Gợi ý OCR chưa lưu (điểm '+scores+'). Nhìn ảnh, sửa nếu sai rồi bấm Lưu nhãn đã xác nhận.')

    def generate_synthetic(self):
        if self.training or self.app.is_busy:return
        try:
            count=int(self.synthetic_count.get())
            if not 1<=count<=100000:raise ValueError('Số ảnh phải từ 1 đến 100.000')
        except ValueError as exc:messagebox.showerror('Số ảnh giả',str(exc),parent=self);return
        self.cancel.clear();self.busy(True);self.bar['value']=0
        seed=time.time_ns() & 0x7fffffff
        excluded={r['plate_id'] for r in self.store.list_examples() if r['split']!='train'}
        self.progress_text.set('Đang sinh dữ liệu giả lập tại máy; chưa huấn luyện trọng số…')
        def worker():
            try:
                from .synthetic import generate_batch
                result=generate_batch(count=count,seed=seed,exclude_plate_ids=excluded,cancel=self.cancel,
                                      progress=lambda event:self.after(0,lambda e=event:self.notify(e)))
                self.after(0,lambda:self.generation_finished(result))
            except Exception as exc:self.after(0,lambda msg=str(exc):self.failed(msg))
        threading.Thread(target=worker,daemon=True).start()

    def generation_finished(self,result):
        self.busy(False);self.refresh()
        if result['completed']:
            self.use_synthetic.set(True);self.bar['value']=100
            self.progress_text.set(f"Đã tạo {result['generated']:,} ảnh giả. Xem mẫu rồi chọn Huấn luyện bổ sung. Thư mục: {result['batch_dir']}")
        else:self.progress_text.set('Đã dừng sinh ảnh. Lô chưa hoàn tất không được đưa vào huấn luyện.')

    def open_synthetic_preview(self):
        from .synthetic import SyntheticStore
        batches=SyntheticStore().batches()
        if not batches:self.message.set('Chưa có lô ảnh giả hoàn tất');return
        root,_=max(batches,key=lambda pair:pair[1]['created_at'])
        os.startfile(str(root/'preview.jpg'))

    def destroy(self):
        self._suggestion_closed=True;self._suggestion_revision+=1
        if self._suggestion_after is not None:
            try:self.after_cancel(self._suggestion_after)
            except tk.TclError:pass
            self._suggestion_after=None
        super().destroy()

    def close(self):
        if self.training:
            self.cancel.set();self.progress_text.set('Đang dừng an toàn; chờ lượt hiện tại hoàn tất rồi đóng cửa sổ.');return
        self.destroy()
