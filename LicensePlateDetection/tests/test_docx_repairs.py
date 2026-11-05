"""Regression checks for geometry, borders and explicitly labelled calibration crops."""
import re
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from image_processing import character_recognition as cr
from image_processing.object_verifier import descriptor, object_margin
from image_processing.plate_verification import ink_layout
from utils.image_utils import crop_license_plate, order_quad_points


class DocumentRegressionTests(unittest.TestCase):
    def test_foreshortened_ordered_quad_preserves_corner_identity(self):
        plate = np.zeros((120, 168, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        plate[:60, :84] = colors[0]
        plate[:60, 84:] = colors[1]
        plate[60:, 84:] = colors[2]
        plate[60:, :84] = colors[3]
        # Projected width is shorter than height: semantic axes must survive.
        q = np.float32([[100, 30], [145, 40], [150, 210], [90, 200]])
        src = np.float32([[0, 0], [167, 0], [167, 119], [0, 119]])
        canvas = cv2.warpPerspective(plate, cv2.getPerspectiveTransform(src, q), (250, 250))
        crop = np.array(crop_license_plate(
            Image.fromarray(canvas), (90, 30, 60, 180), margin_ratio=0,
            quad_corners=q, ordered_quad=True, target_aspect_ratio=1.4))
        self.assertAlmostEqual(crop.shape[1] / crop.shape[0], 1.4, delta=.01)
        for position, expected in zip([(.25, .25), (.25, .75), (.75, .75), (.75, .25)], colors):
            y, x = position
            np.testing.assert_allclose(crop[int(y*crop.shape[0]), int(x*crop.shape[1])], expected, atol=5)

    def test_unordered_quad_is_convex_for_quarter_turn(self):
        points = np.float32([[60, 10], [100, 20], [80, 200], [40, 190]])
        ordered = order_quad_points(points[[2, 0, 3, 1]])
        edges = np.roll(ordered, -1, axis=0)-ordered
        crosses = edges[:, 0]*np.roll(edges[:, 1], -1)-edges[:, 1]*np.roll(edges[:, 0], -1)
        self.assertTrue(np.all(crosses > 0) or np.all(crosses < 0))

    def test_frame_filter_keeps_real_one_on_baseline(self):
        mask = np.zeros((120, 180), bool)
        boxes = [(4, 5, 8, 24), (35, 18, 8, 42), (65, 18, 20, 42), (105, 18, 20, 42), (145, 18, 8, 42)]
        row = []
        for x, y, w, h in boxes:
            mask[y:y+h, x:x+w] = True
            row.append({'bbox': (x, y, w, h), 'patch': np.ones((h, w), bool), 'cx': x+w/2, 'cy': y+h/2})
        repaired = cr._repair_row(row, mask, cr.get_template_matcher(), False)
        self.assertEqual([c['bbox'] for c in repaired], boxes[1:])

    def test_format_check_does_not_accept_letters_in_numeric_body(self):
        for text in ['18-MD2 074.73', '98-N6 6762', '34-A 789.22', '29A-146.08']:
            self.assertTrue(cr.is_valid_vn_plate(text), text)
        for text in ['43-H1 L838', '89-ABT 142.94', '29-G14 727.58', '30-4 8044']:
            self.assertFalse(cr.is_valid_vn_plate(text), text)

    def test_red_car_consensus_ignores_overlapping_frame_glyph(self):
        image = Image.open(BASE/'tests/fixtures/docx_red_car_crop.png').convert('RGB')
        result = cr.recognize_license_plate(image, 'double_row', True, backend='template')
        self.assertEqual(re.sub(r'[^A-Z0-9]', '', result['plate_text']), '34A78922')
        self.assertEqual(result['char_count'], 8)
        self.assertTrue(result['needs_review'])

    def test_nonstandard_truck_prefix_and_touching_sixty_are_preserved(self):
        image = Image.open(BASE/'tests/fixtures/docx_truck_crop.png').convert('RGB')
        result = cr.recognize_license_plate(image, 'double_row', True, backend='template')
        self.assertEqual(re.sub(r'[^A-Z0-9]', '', result['plate_text']), 'T6167960')
        self.assertTrue(result['needs_review'])
        self.assertEqual(sum(c['row']==2 for c in result['characters']), 5)

    def test_rejected_proposals_still_save_current_image_preview(self):
        import tempfile
        from unittest.mock import patch
        from main import process_single_image
        from utils.file_utils import create_run_folders
        (BASE/'output').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='regression_', dir=BASE/'output') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((BASE/'output').resolve()))
            path = Path(folder)/'blank.png'
            Image.new('RGB', (100, 80), 'white').save(path)
            dirs = create_run_folders(Path(folder)/'out', run_name='test')
            candidate = {'bbox': (10, 10, 40, 30), 'source': 'boundary',
                         'plate_type': 'double_row', 'score': .9, 'angle': 0.}
            with patch('image_processing.plate_verification.boundary_candidates', return_value=[candidate]):
                rows = process_single_image(path, dirs, save_intermediate=False)
            self.assertEqual(rows[0]['plate_index'], 0)
            self.assertTrue((dirs['detected']/'blank_detected.png').is_file())

    def test_user_confirmed_eight_is_not_read_as_six(self):
        image = Image.open(BASE/'tests/fixtures/docx_29g1_user_confirmed.png').convert('RGB')
        result = cr.recognize_license_plate(image, 'double_row', True, backend='template')
        self.assertEqual(re.sub(r'[^A-Z0-9]', '', result['plate_text']), '29G172758')
        self.assertEqual(result['char_count'], 9)

    def test_common_border_and_object_calibration_regressions(self):
        # These are calibration regressions, not an independent accuracy benchmark.
        data = BASE/'data/plate_verifier'
        for file in ['negative/reviewed_1793454835379.png', 'negative/reviewed_1793454835388.png', 'negative/reviewed_1793454835397.png']:
            gray = cv2.imread(str(data/file), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(descriptor(gray).shape, (865,))
            self.assertIsNotNone(object_margin(gray), file)
        n6 = cv2.imread(str(data/'positive/09.png'), cv2.IMREAD_GRAYSCALE)
        n6 = cv2.resize(n6, (round(n6.shape[1]*120/n6.shape[0]), 120))
        self.assertIsNotNone(ink_layout(n6, 'double_row'))


if __name__ == '__main__':
    unittest.main()
