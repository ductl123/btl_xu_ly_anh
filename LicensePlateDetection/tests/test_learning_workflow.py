"""Meaningful checks for supervised data, CTC, training and model promotion."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from learning.dataset import DatasetStore,clean_label,verify_splits,image_digest
from learning.line_model import MODEL_DIR,ALPHABET,canonical_log_probs,decode,load_model,predict_lines,recognize
from learning.trainer import train
from learning.restoration import wiener_restore


class LearningWorkflowTests(unittest.TestCase):
    def setUp(self):
        (BASE/'output').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='learning_test_',dir=BASE/'output')
        self.root=Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to((BASE/'output').resolve()))
        self.store=DatasetStore(self.root/'data')

    def tearDown(self):self.temp.cleanup()

    def example(self,color='white'):
        image=Image.new('RGB',(260,120),color)
        draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',36)
        draw.text((30,5),'18-C1',font=font,fill='black');draw.text((20,62),'277.11',font=font,fill='black')
        return image

    def test_labels_are_explicit_and_separators_do_not_become_classes(self):
        self.assertEqual(clean_label(' 18-c1 '),'18C1')
        self.assertEqual(clean_label('277.11'),'27711')
        for invalid in ['','???','18CỐ','18/O']:
            with self.assertRaises(ValueError):clean_label(invalid)

    def test_duplicate_image_is_not_counted_as_extra_data(self):
        image=self.example()
        first=self.store.save(image,'18C1','27711')
        self.assertEqual(first,self.store.save(image,'18-C1','277.11'))
        self.assertEqual(len(self.store.list_examples()),1)

    def test_same_vehicle_cannot_cross_training_and_validation(self):
        self.store.save(self.example(),'18C1','27711',split='train')
        with self.assertRaises(ValueError):
            self.store.save(self.example('yellow'),'18C1','27711',split='validation')

    def test_same_scene_with_different_plates_cannot_cross_splits(self):
        self.store.save(self.example(),'18C1','27711',source_group='photo_a')
        with self.assertRaises(ValueError):
            self.store.save(self.example('yellow'),'34B4','76792',source_group='photo_a',split='test')

    def test_removing_label_keeps_the_image_and_excludes_it_from_training(self):
        ident=self.store.save(self.example(),'18C1','27711')
        record=self.store.list_examples()[0];path=self.store.image_path(record)
        self.store.remove(ident)
        self.assertEqual(self.store.list_examples(),[]);self.assertTrue(path.is_file())

    def test_ctc_treats_dash_as_blank_and_merges_letter_case(self):
        charset=json.loads((MODEL_DIR/'charset.json').read_text())
        logits=torch.full((1,5,len(charset)+1),-20.)
        for t,char in enumerate(['3','-','B','b','.']):logits[0,t,charset.index(char)+1]=20.
        result=decode(canonical_log_probs(logits,charset)[0],5)
        self.assertEqual(''.join(c['char'] for c in result),'3B')

    def test_wiener_recovers_a_known_mild_gaussian_blur(self):
        original=np.full((80,100),220,np.uint8)
        original[20:60,20:35]=30;original[20:60,65:80]=30;original[38:45,20:80]=30
        blurred=cv2.GaussianBlur(original,(5,5),1.)
        restored=wiener_restore(blurred)
        mse=lambda a:np.mean((a.astype(float)[8:-8,8:-8]-original[8:-8,8:-8])**2)
        self.assertLess(mse(restored),mse(blurred))
        self.assertLessEqual(np.max(np.abs(wiener_restore(np.full((30,30),120,np.uint8)).astype(int)-120)),1)

    def test_actual_gradient_updates_save_and_reload_while_test_data_is_excluded(self):
        models=self.root/'models';models.mkdir()
        for name in ['english_g2.pth','charset.json','pretrained_metadata.json']:
            shutil.copy2(MODEL_DIR/name,models/name)
        self.store.save(self.example(),'18C1','27711',source_group='train_a')
        self.store.save(self.example('yellow'),'37AC','17124',source_group='val_a',split='validation')
        test_id=self.store.save(self.example('lightgray'),'98E1','83445',source_group='test_a',split='test')
        good={'character_error_rate':0.,'correct':1,'plates':1}
        bad={'character_error_rate':.5,'correct':0,'plates':1}
        # Baseline(val,train), then epoch1 worse val, then epoch2 same val and
        # improved train. The first candidate must never replace the baseline.
        events=[]
        with patch('learning.trainer.evaluate',side_effect=[good,bad,bad,good,good,good]):
            result=train(self.store,models,epochs=2,batch_size=1,progress=events.append)
        self.assertGreater(result['head_weight_change_l2'],0.)
        self.assertEqual(result['best_epoch'],2);self.assertTrue(result['activated'])
        self.assertIn(test_id,result['test_ids_excluded'])
        self.assertNotIn(test_id,result['training_ids']);self.assertNotIn(test_id,result['validation_ids'])
        before,_,_,_=load_model(models,checkpoint=models/'english_g2.pth')
        after,chars,device,version=load_model(models)
        self.assertFalse(torch.equal(before.Prediction.weight,after.Prediction.weight))
        self.assertEqual(version,result['version'])

    def test_training_requires_a_separate_validation_set(self):
        self.store.save(self.example(),'18C1','27711')
        with self.assertRaisesRegex(ValueError,'Kiểm định'):
            train(self.store,self.root/'models',epochs=1)

    def test_no_validation_improvement_preserves_current_published_model(self):
        models=self.root/'models';models.mkdir()
        for name in ['english_g2.pth','charset.json','pretrained_metadata.json']:shutil.copy2(MODEL_DIR/name,models/name)
        self.store.save(self.example(),'18C1','27711')
        self.store.save(self.example('yellow'),'37AC','17124',split='validation')
        good={'character_error_rate':0.,'correct':1,'plates':1};bad={'character_error_rate':.5,'correct':0,'plates':1}
        with patch('learning.trainer.evaluate',side_effect=[good,bad,bad,good]):
            result=train(self.store,models,epochs=1,batch_size=1)
        self.assertFalse(result['activated']);self.assertFalse((models/'active.json').exists())
        self.assertTrue(Path(result['last_checkpoint']).is_file())

    def test_manual_annotation_window_can_save_and_reload_a_label(self):
        import tkinter as tk
        from learning.training_ui import TrainingWindow
        root=tk.Tk();root.withdraw()
        class App:
            is_busy=False;current_candidates=[];current_image_path=None
            tree=type('Tree',(),{'selection':lambda s:[]})()
            def _set_buttons_state(self,value):pass
        app=App();app.root=root
        try:
            with patch('learning.training_ui.DatasetStore',return_value=self.store):window=TrainingWindow(app)
            window.withdraw();window.image=self.example();window.preview()
            original=image_digest(window.image)
            for _ in range(4):window.rotate_image()
            self.assertEqual(image_digest(window.image),original)
            window.row1.set('18-C1');window.row2.set('277.11');window.save()
            self.assertEqual(self.store.list_examples()[0]['plate_id'],'18C127711')
            window.destroy()
        finally:root.destroy()


if __name__=='__main__':unittest.main()
