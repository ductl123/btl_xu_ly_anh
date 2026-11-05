import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_processing import character_recognition as cr

class OCRRegression(unittest.TestCase):
    def test_own_templates_are_not_corrupted(self):
        matcher = cr.get_template_matcher()
        for char, templates in matcher.templates.items():
            for template in templates[::15] + templates[-3:]:
                with self.subTest(char=char):
                    result = matcher.match_details(template, char.isdigit(), not char.isdigit())
                    self.assertEqual(result['char'], char)
                    self.assertLessEqual(result['score'], 1.0)

    def test_clear_plate_does_not_change_two_to_three(self):
        img = Image.new('RGB', (420, 90), 'white')
        font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 48)
        ImageDraw.Draw(img).text((18, 10), '30A12345', fill='black', font=font)
        result = cr.recognize_license_plate(img, 'single_row', backend='template')
        self.assertEqual(re.sub(r'[^A-Z0-9]', '', result['plate_text']), '30A12345')
        self.assertEqual(result['char_count'], 8)
        for char in result['characters']:
            self.assertIn('top_candidates', char['match_details'])

    def test_sauvola_fallback_keeps_black_ink(self):
        image = Image.new('RGB', (200, 120), (200, 200, 200))
        ImageDraw.Draw(image).rectangle((50, 30, 69, 89), fill='black')
        captured = []
        def capture(mask, **kwargs):
            captured.append(mask.copy())
            return []
        with patch.object(cr, 'segment_characters', side_effect=capture):
            cr._recognize_single_orientation(image, 'single_row')
        sauvola = captured[1]
        self.assertTrue(sauvola[40:80, 55:65].all())
        self.assertFalse(sauvola[10:20, 100:110].any())
        self.assertFalse(sauvola[0, :].any())

if __name__ == '__main__':
    unittest.main()
