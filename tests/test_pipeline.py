import argparse
import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from roadsense.pipeline import audit_rows, damage_types, fingerprints, index, read_csv, split_rows

class PipelineTests(unittest.TestCase):
    def test_index_xml_unreadable_and_dry_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'images'; root.mkdir()
            Image.new('RGB', (10, 10), 'red').save(root / 'a.jpg')
            (root / 'a.xml').write_text('<annotation><object><name>D00</name></object><object><name>D40</name></object></annotation>')
            (root / 'bad.jpg').write_text('bad')
            out = Path(d) / 'index.csv'; errors = Path(d) / 'errors.csv'
            args = argparse.Namespace(root=str(root), source='rdd', manifest=None, annotations=None, all=False, limit=100, dry_run=True, output=str(out), errors=str(errors))
            index(args); self.assertFalse(out.exists())
            args.dry_run = False; index(args)
            rows = read_csv(out)
            self.assertEqual(rows[0]['damage_types'], 'D00|D40')
            self.assertEqual(rows[0]['quality_label'], '')
            self.assertEqual(len(read_csv(errors)), 1)
    def test_audit_and_split_reproducible_group_safe(self):
        rows = []
        for i in range(20):
            rows.append(dict(image_id=str(i), source='local', group_id=str(i//2), country='Cambodia', quality_label='Good', sha256=str(i), dhash=f'{i*1000000000000:016x}'))
        rows[1]['sha256'] = rows[2]['sha256']
        a = split_rows(rows, 42, .2, 0); b = split_rows(rows, 42, .2, 0)
        self.assertEqual(a, b)
        self.assertEqual(a[1]['split'], a[2]['split'])
        for i in range(0, 20, 2): self.assertEqual(a[i]['split'], a[i+1]['split'])
        self.assertTrue(any(r['split'] == 'cambodia_holdout' for r in a))
        self.assertEqual(audit_rows(rows, 0)['exact_duplicates'], [['1', '2']])
    def test_requires_independent_cambodian_groups(self):
        with self.assertRaises(ValueError):
            split_rows([dict(image_id='1', source='rdd', group_id='', country='', quality_label='Good', sha256='', dhash='')], 42, .2, 4)

if __name__ == '__main__': unittest.main()
