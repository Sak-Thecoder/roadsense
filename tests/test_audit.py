import tempfile, unittest
from pathlib import Path
from PIL import Image
from roadsense.audit import process, group_duplicates, pilot
from roadsense.prepare import prepare

class AuditTests(unittest.TestCase):
    def test_xml_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'Japan_001.jpg'; Image.new('RGB',(12,8),'blue').save(p)
            p.with_suffix('.xml').write_text('<annotation><object><name>D00</name></object><object><name>D40</name></object></annotation>')
            row,err=process((root,p)); self.assertEqual(row['damage_types'],'D00|D40'); self.assertEqual(row['quality_label'],''); self.assertFalse(err)
            p.with_suffix('.xml').unlink(); self.assertEqual(process((root,p))[0]['annotation_status'],'missing')
    def test_duplicate_and_pilot(self):
        rows=[dict(image_id=str(i),sha256=str(i),dhash=f'{i:016x}',duplicate_group='',country='Japan',damage_types='D00',annotation_status='annotated') for i in range(5)]
        group_duplicates(rows); self.assertEqual(len(pilot(rows,5,42)),1)
    def test_review_and_split_group(self):
        rows=[dict(image_id=str(i),duplicate_group='same' if i<2 else '',scene_group='',country='Japan') for i in range(6)]
        labels=[dict(image_id=str(i),quality_label='Good') for i in range(6)]
        result,summary=prepare(rows,labels,42,3); assignments={r['image_id']:r['split'] for r in result}
        self.assertEqual(assignments['0'],assignments['1']); self.assertEqual(summary['reviewed'],6); self.assertEqual(summary['shortages_to_minimum']['Poor'],3)
        self.assertFalse(summary['cambodian_holdout_created'])
if __name__=='__main__': unittest.main()
