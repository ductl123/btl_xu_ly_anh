"""Generated ground truth, held-out isolation and human confirmation checks."""
from pathlib import Path
import json,random,shutil,sys,tempfile,threading,unittest
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image,ImageDraw,ImageFont

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from learning.synthetic import generate_batch,SyntheticStore,make_text
from learning.dataset import ALPHABET,DatasetStore,clean_label


class SyntheticWorkflowTests(unittest.TestCase):
    def setUp(self):
        (BASE/'output').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic_test_',dir=BASE/'output')
        self.root=Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to((BASE/'output').resolve()))

    def tearDown(self):self.temp.cleanup()

    def test_labels_images_and_all_30_classes_match_and_repeat_is_idempotent(self):
        first=generate_batch(self.root,count=128,seed=17)
        second=generate_batch(self.root,count=128,seed=17)
        self.assertFalse(first['reused']);self.assertTrue(second['reused'])
        store=SyntheticStore(self.root);records=store.records()
        self.assertEqual(len(records),128);self.assertEqual(store.count(),128)
        self.assertEqual(set(''.join(r['text'] for r in records)),set(ALPHABET))
        self.assertTrue(any('.' in r['rendered_text'] for r in records))
        for record in records:
            image,label=store.load(record)
            self.assertEqual(clean_label(record['rendered_text']),label)
            self.assertEqual(image.height,64);self.assertGreaterEqual(image.width,48)
            self.assertEqual(record['split'],'train')
        heldout=records[0]['plate_id']
        self.assertNotIn(heldout,[r['plate_id'] for r in store.records([heldout])])

    def test_cancelled_batch_is_not_used_and_can_resume_deterministically(self):
        stop=threading.Event();stop.set()
        cancelled=generate_batch(self.root,count=2,seed=18,cancel=stop)
        self.assertFalse(cancelled['completed']);self.assertEqual(SyntheticStore(self.root).count(),0)
        completed=generate_batch(self.root,count=2,seed=18)
        self.assertTrue(completed['completed']);self.assertEqual(SyntheticStore(self.root).count(),2)

    def test_holdout_exclusion_changes_batch_identity_and_image_tampering_is_rejected(self):
        plate=make_text(0,random.Random(99^0x9e3779b97f4a7c15))['plate_id']
        first=generate_batch(self.root,count=2,seed=99)
        second=generate_batch(self.root,count=2,seed=99,exclude_plate_ids=[plate])
        self.assertNotEqual(first['batch_id'],second['batch_id'])
        store=SyntheticStore(self.root);record=store.records()[0]
        (Path(record['batch_root'])/record['image']).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError,'khớp nhãn'):store.load(record)
        record['image']='../../outside.jpg'
        with self.assertRaisesRegex(ValueError,'path'):store.load(record)

    def test_manifest_corruption_and_invalid_counts_are_rejected(self):
        batch=generate_batch(self.root,count=2)
        (Path(batch['batch_dir'])/'manifest.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'Manifest'):SyntheticStore(self.root).records()
        for count in [0,100001]:
            with self.assertRaises(ValueError):generate_batch(self.root,count=count)

    def test_gui_drafts_never_save_or_overwrite_edits_or_apply_to_another_image(self):
        import tkinter as tk
        from learning.training_ui import TrainingWindow
        root=tk.Tk();root.withdraw();store=DatasetStore(self.root/'real')
        app=SimpleNamespace(root=root,is_busy=False,current_candidates=[],current_image_path=None,
                            tree=SimpleNamespace(selection=lambda:[]),_set_buttons_state=lambda _:None)
        try:
            with patch('learning.training_ui.DatasetStore',return_value=store):window=TrainingWindow(app)
            window.withdraw();window.image=Image.new('RGB',(300,150),'white');window.preview()
            token=window._suggestion_revision
            draft={'row1':'18C1','row2':'77711','scores':[.9,.8],'status':'unconfirmed_draft','model_version':'test'}
            window._apply_suggestion(token,('',''),draft)
            self.assertEqual(window.row1.get(),'18C1');self.assertEqual(store.list_examples(),[])
            window.row2.set('27711')
            window._apply_suggestion(token,('',''),draft)
            self.assertEqual(window.row2.get(),'27711')
            window._apply_suggestion(token-1,('',''),{**draft,'row1':'WRONG'})
            self.assertEqual(window.row1.get(),'18C1')
            window.save();record=store.list_examples()[0]
            self.assertEqual(record['plate_id'],'18C127711')
            self.assertEqual(json.loads(record['provenance'])['ocr_draft']['status'],'unconfirmed_draft')
            window.destroy()
        finally:root.destroy()

    def test_mixed_training_updates_weights_excludes_test_and_limits_synthetic_consumption(self):
        from learning.trainer import train
        from learning.line_model import MODEL_DIR
        models=self.root/'models';models.mkdir();store=DatasetStore(self.root/'real')
        for name in ['english_g2.pth','charset.json','pretrained_metadata.json']:shutil.copy2(MODEL_DIR/name,models/name)
        for i,split in enumerate(['train','validation','test']):
            image=Image.new('RGB',(260,120),(220+i*10,220,220));draw=ImageDraw.Draw(image)
            font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',36)
            draw.text((20,3),'18-C1',font=font,fill='black');draw.text((20,62),f'277.1{i}',font=font,fill='black')
            store.save(image,'18C1',f'2771{i}',split=split)
        synthetic=self.root/'synthetic';generate_batch(synthetic,count=8,seed=10)
        good={'character_error_rate':0.,'correct':1,'plates':1}
        bad={'character_error_rate':.5,'correct':0,'plates':1}
        # Actual forward/backprop and synthetic evaluations; only mock real-set
        # ranking so this small test does not depend on pretrained OCR accuracy.
        with patch('learning.trainer.evaluate',side_effect=[good,bad,good,good]):
            report=train(store,models,epochs=1,batch_size=4,use_synthetic=True,
                         synthetic_root=synthetic,synthetic_per_epoch=4)
        self.assertGreater(report['head_weight_change_l2'],0.)
        self.assertEqual(report['synthetic_unique_used'],4)
        self.assertEqual(report['synthetic_available'],8)
        self.assertEqual(report['baseline_synthetic_training']['lines'],8)
        self.assertTrue(report['activated'])
        self.assertEqual(len(report['test_ids_excluded']),1)
        self.assertFalse(set(report['test_ids_excluded'])&set(report['training_ids']))
        self.assertEqual(report['trained_character_counts'].get('7'),2)


if __name__=='__main__':unittest.main()
