"""Apply a corrected Cambodian review CSV into a new unified manifest version."""
import argparse,csv,json,datetime
from pathlib import Path

def read(p):
 with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows,fields):
 with open(p,'x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def apply(review,rows):
 records={r['image_id']:r for r in review}
 if len(records)!=100 or len(review)!=100:raise ValueError('Review file must have 100 unique IDs')
 matched=0
 for row in rows:
  if row['source']!='RoadSense_Cambodian':continue
  c=records.get(row['original_id'])
  if c is None:raise ValueError('Missing '+row['original_id'])
  if c['review_label'] not in ('Good','Moderate','Poor','Uncertain','Reject') or c['review_status'] not in ('reviewed','unreviewed'):raise ValueError('Invalid review label or status')
  if c['review_date']:
   try: datetime.date.fromisoformat(c['review_date'])
   except ValueError: raise ValueError('Review date must be YYYY-MM-DD')
  if c['source']=='cambodia_online' and c['training_permission']=='confirmed' and not c.get('rights_notes','').strip(): raise ValueError('Online permission confirmation requires evidence in rights_notes')
  row.update(verified_quality_label=c['review_label'] if c['review_status']=='reviewed' else '',label_status=c['review_status'],reviewer=c['reviewer'],review_date=c['review_date'],capture_group=c['group_id'],permission_status=c['training_permission'],source_metadata=json.dumps(c,ensure_ascii=False,sort_keys=True))
  reasons=[]
  if c['review_status']!='reviewed' or c['review_label'] not in ('Good','Moderate','Poor'):reasons.append('not_reviewed_three_class')
  if c['training_permission']!='confirmed':reasons.append('training_permission_unconfirmed')
  if not c['reviewer'] or not c['review_date']:reasons.append('reviewer_or_date_missing')
  if not c['group_id']:reasons.append('capture_group_unverified')
  if row.get('duplicate_group'):reasons.append('duplicate_candidate_review')
  if c['image_id']=='CAM023':reasons.append('exact_duplicate_of_CAM022')
  row['exclusion_reasons']='|'.join(reasons);row['eligible']=str(not reasons);matched+=1
 if matched!=100:raise ValueError('Unified metadata must have 100 Cambodian records')
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--unified',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():p.error('Choose a new output directory')
 review=read(a.review);rows=apply(review,read(a.unified));out.mkdir(parents=True);write(out/'unified_metadata.csv',rows,list(rows[0]));write(out/'training_ready.csv',[r for r in rows if r['eligible'].lower()=='true'],list(rows[0]));print('Updated Cambodian records:',len(review));print('Eligible:',sum(r['eligible'].lower()=='true' for r in rows))
if __name__=='__main__':main()
