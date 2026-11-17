import csv, json, unittest, importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def load(name):
 spec=importlib.util.spec_from_file_location(name,P/'scripts'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
class BaselineTests(unittest.TestCase):
 @unittest.skipUnless(all((P/'manifests'/name).is_file() for name in ('cambodian_review_working.csv','unified_current.csv')),
                      'Requires the local Cambodian and unified research manifests')
 def test_approval_required_and_separate_online_rights(self):
  m=load('sync_cambodian');cam=read(P/'manifests/cambodian_review_working.csv');unified=read(P/'manifests/unified_current.csv')
  with self.assertRaises(ValueError):m.sync(cam,unified,{})
  cam,unified=m.sync(cam,unified,{'approve_recorded_95':True,'authorize_self_collected_50':True})
  self.assertEqual(sum(r['review_status']=='reviewed' for r in cam),100)
  self.assertTrue(all(r['training_permission']=='unconfirmed' for r in cam if r['source']=='cambodia_online'))
  self.assertTrue(all(r['training_permission']=='confirmed' for r in cam if r['source']=='self_collected'))
  dup=next(r for r in unified if r['original_id']=='CAM023');self.assertIn('exact_duplicate_of_CAM022',dup['exclusion_reasons'])
  self.assertFalse(any(r['eligible']=='True' for r in unified if r['source']=='RoadSense_Cambodian'))
 @unittest.skipUnless((P/'manifests/street_provisional.csv').is_file(),
                      'Requires the local StreetSurfaceVis research manifest')
 def test_street_mapping_unverified_and_test_preserved(self):
  rows=read(P/'manifests/street_provisional.csv');self.assertEqual(len(rows),9122)
  self.assertEqual(sum(r['original_split']=='test' for r in rows),776)
  self.assertTrue(all(not r['verified_quality_label'] for r in rows))
  self.assertEqual(sum(r['proposed_quality_label']=='Good' for r in rows),4908)
if __name__=='__main__':unittest.main()
