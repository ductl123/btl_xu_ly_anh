import sys,unittest,json
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b))
from image_processing.candidate_filter import refine_quad_corners
from utils.image_utils import crop_license_plate,order_quad_points

class RectificationTests(unittest.TestCase):
 def test_tilted_and_perspective_boundaries_at_two_scales(self):
  plate=Image.new('RGB',(320,220),'white');d=ImageDraw.Draw(plate)
  font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',60)
  d.text((30,10),'37-AC',font=font,fill='black');d.text((18,105),'17124',font=font,fill='black')
  src=np.float32([[0,0],[319,0],[319,219],[0,219]])
  for pts in [[[110,130],[440,50],[485,290],[160,360]],[[70,70],[440,130],[380,370],[110,320]],[[95,95],[425,95],[425,325],[95,325]]]:
   q=np.float32(pts);mat=cv2.getPerspectiveTransform(src,q)
   canvas=cv2.warpPerspective(np.array(plate),mat,(560,440),borderValue=(30,30,30))
   for scale in [1.,.5]:
    found,angle=refine_quad_corners(canvas,tuple(np.array([40,30,480,370])*scale),scale)
    self.assertIsNotNone(found)
    np.testing.assert_allclose(found/scale,order_quad_points(q),atol=18)
    crop=crop_license_plate(Image.fromarray(canvas),(40,30,480,370),quad_corners=found/scale,margin_ratio=.02)
    self.assertTrue(1.15<crop.width/crop.height<1.8)

if __name__=='__main__':unittest.main()
