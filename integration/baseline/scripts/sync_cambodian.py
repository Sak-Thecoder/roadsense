"""Synchronize an explicitly confirmed Cambodian review export into versioned manifests."""
import argparse,csv,datetime,json
from collections import Counter
from pathlib import Path

def read(path):
 with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows,fields):
 with open(path,'x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def sync(base,old,decision):
 if not decision.get('approve_recorded_95'):raise ValueError('Explicit approval of recorded labels required')
 if len(base)!=100 or len({r['image_id'] for r in base})!=100:raise ValueError('Expected 100 unique Cambodian records')
 if Counter(r['review_status'] for r in base)!=Counter({'reviewed':5,'unreviewed':95}):raise ValueError('Review statuses differ from confirmed snapshot')
 pending=Counter(r['review_label'] for r in base if r['review_status']=='unreviewed')
 if pending!=Counter({'Good':39,'Moderate':40,'Poor':15,'Uncertain':1}):raise ValueError('Label values differ from confirmed snapshot')
 if sum(r['source']=='self_collected' for r in base)!=50:raise ValueError('Self-collected count differs from confirmed snapshot')
 for r in base:
  if r['review_status']=='unreviewed':r['review_status']='reviewed';r['review_notes']=(r.get('review_notes','')+'; dataset owner confirmed previously recorded label; review date and reviewer name not supplied').strip('; ')
  if r['source']=='self_collected' and decision.get('authorize_self_collected_50'):
   r['training_permission']='confirmed';r['rights_notes']=(r.get('rights_notes','')+'; dataset owner confirmed ownership and RoadSense training use').strip('; ')
 byid={r['image_id']:r for r in base}; used=0
 for r in old:
  if r['source']!='RoadSense_Cambodian':continue
  c=byid.get(r['original_id'])
  if not c:raise ValueError('Missing Cambodian record '+r['original_id'])
  r['verified_quality_label']=c['review_label'];r['label_status']=c['review_status'];r['reviewer']=c['reviewer'];r['review_date']=c['review_date'];r['permission_status']=c['training_permission'];r['source_metadata']=json.dumps(c,ensure_ascii=False,sort_keys=True)
  reasons=[]
  if c['review_label'] not in ('Good','Moderate','Poor'):reasons.append('not_three_class')
  if c['training_permission']!='confirmed':reasons.append('training_permission_unconfirmed')
  if not c['reviewer'] or not c['review_date']:reasons.append('reviewer_or_date_missing')
  if not c['group_id']:reasons.append('capture_group_unverified')
  if r.get('duplicate_group'):reasons.append('duplicate_candidate_review')
  if c['image_id']=='CAM023':reasons.append('exact_duplicate_of_CAM022')
  r['exclusion_reasons']='|'.join(reasons);r['eligible']=str(not reasons);used+=1
 if used!=100:raise ValueError('Unified metadata did not contain 100 Cambodian records')
 return base,old

def main():
 p=argparse.ArgumentParser();p.add_argument('--cambodian',required=True);p.add_argument('--unified',required=True);p.add_argument('--confirmation',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():p.error('Output already exists; choose a new version')
 base,rows=sync(read(a.cambodian),read(a.unified),json.loads(Path(a.confirmation).read_text()));out.mkdir(parents=True)
 write(out/'cambodian_reviewed.csv',base,list(base[0]));write(out/'unified_metadata.csv',rows,list(rows[0]));write(out/'training_ready.csv',[r for r in rows if str(r['eligible']).lower()=='true'],list(rows[0]));print(json.dumps({'reviewed':Counter(r['review_status'] for r in base),'verified_labels':Counter(r['review_label'] for r in base),'permission_by_source':Counter((r['source'],r['training_permission']) for r in base).__str__(),'cambodian_ready':sum(r['source']=='RoadSense_Cambodian' and str(r['eligible']).lower()=='true' for r in rows)},indent=2))
if __name__=='__main__':main()
