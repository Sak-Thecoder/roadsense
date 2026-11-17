import csv, subprocess, sys, tempfile, unittest, zipfile
from pathlib import Path
from PIL import Image

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'audit_integration.py'
class IntegrationTests(unittest.TestCase):
 def test_manifest_review_permissions_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'data'; rdd=root/'RDD_SPLIT/train'; (rdd/'images').mkdir(parents=True); (rdd/'labels').mkdir()
   img=rdd/'images/Japan_001.jpg';Image.new('RGB',(15,15),'red').save(img);(rdd/'labels/Japan_001.txt').write_text('0 .5 .5 .1 .1\n')
   street=root/'streetSurfaceVis'; (street/'s_1024').mkdir(parents=True);Image.new('RGB',(15,15),'blue').save(street/'s_1024/1.jpg')
   (street/'streetSurfaceVis_v1_0.csv').write_text('mapillary_image_id,train,surface_quality,user_name\n1,False,good,A\n')
   cam=root/'RoadSense_Cambodian';cam.mkdir();
   with zipfile.ZipFile(cam/'cambodia_road.zip','w') as z:z.write(img,'photos/a.jpg')
   with (cam/'dataset_manifest.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path_in_archive','image_id','review_status','review_label','training_permission']);w.writeheader();w.writerow(dict(path_in_archive='photos/a.jpg',image_id='1',review_status='reviewed',review_label='Poor',training_permission='unconfirmed'))
   out=Path(d)/'out';cmd=[sys.executable,str(SCRIPT),'--dataset-root',str(root),'--output',str(out)]
   run=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
   rows=list(csv.DictReader((out/'unified_metadata.csv').open()));self.assertEqual(len(rows),3)
   s=next(r for r in rows if r['source']=='streetSurfaceVis');self.assertEqual(s['original_split'],'non_train');self.assertEqual(s['proposed_quality_label'],'Good');self.assertEqual(s['verified_quality_label'],'')
   c=next(r for r in rows if r['source']=='RoadSense_Cambodian');self.assertEqual(c['verified_quality_label'],'Poor');self.assertEqual(c['eligible'],'False')
   self.assertEqual(len(list(csv.DictReader((out/'training_ready.csv').open()))),0)
   self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
if __name__=='__main__':unittest.main()
