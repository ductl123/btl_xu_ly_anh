import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_processing.character_recognition import get_template_matcher, compute_euler_holes, binarize_cropped_plate, segment_characters
import cv2

class CommonMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = get_template_matcher()

    def test_empty_is_unknown_not_an_arbitrary_character(self):
        result = self.matcher.match_details(np.zeros((32,20),bool))
        self.assertEqual(result['char'], '?')
        self.assertTrue(result['needs_review'])

    def test_invalid_restrictions_are_rejected(self):
        with self.assertRaises(ValueError):
            self.matcher.match_details(np.ones((32,20)), True, True)

    def test_30_classes_are_present_and_group_restrictions_hold(self):
        self.assertEqual(len(self.matcher.templates), 30)
        for char, masks in self.matcher.templates.items():
            self.assertTrue(masks)
            mask = masks[1]
            result = self.matcher.match_details(mask, char.isdigit(), not char.isdigit())
            self.assertEqual(result['char'], char)
            self.assertTrue(all(c['char'].isdigit()==char.isdigit() for c in result['top_candidates']))
            self.assertTrue(0 <= result['score'] <= 1)

    def test_open_c_is_not_invented_as_one_hole(self):
        arr = np.zeros((32,20),dtype=bool)
        arr[2:30,2:6]=True
        arr[2:6,2:17]=True
        arr[26:30,2:17]=True
        self.assertEqual(compute_euler_holes(arr), 0)
        arr[2:30,13:17]=True
        self.assertEqual(compute_euler_holes(arr), 1)

    def test_sparse_thin_ink_does_not_turn_background_white(self):
        image = Image.new('L', (100,100), 230)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 56)
        ImageDraw.Draw(image).text((15,5), '1', font=font, fill=20)
        gray = cv2.dilate(np.array(image), np.ones((2,2),np.uint8))
        mask = binarize_cropped_plate(Image.fromarray(gray))
        self.assertTrue(.001 < mask.mean() < .20)
        ink = gray < 100
        self.assertGreater((mask & ink).sum()/ink.sum(), .8)

    def test_character_patch_does_not_include_unconnected_dust(self):
        mask = np.zeros((100,100), dtype=bool)
        mask[20:80,20:25] = True
        mask[20:80,45:50] = True
        mask[75:80,20:50] = True
        mask[45:47,34:36] = True  # unrelated dust inside the U's bounding box
        chars = segment_characters(mask, 'single_row')
        glyph = next(c for c in chars if c['bbox'] == (20,20,30,60))
        self.assertFalse(glyph['patch'][25:27,14:16].any())
        self.assertTrue(glyph['patch'][:,0].all())

    def test_cache_cannot_be_changed_by_a_caller(self):
        mask = self.matcher.templates['3'][1]
        result = self.matcher.match_details(mask, True)
        result['top_candidates'][0]['char'] = 'X'
        again = self.matcher.match_details(mask, True)
        self.assertEqual(again['char'], '3')
        self.assertNotEqual(again['top_candidates'][0]['char'], 'X')

if __name__ == '__main__':
    unittest.main()
