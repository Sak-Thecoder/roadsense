"""Read source datasets and create a non-destructive integration audit."""
import argparse, collections, csv, hashlib, io, json, os, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from roadsense.audit import group_duplicates

EXT={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
FIELDS=['image_id','source','original_id','image_path','archive_member','annotation_path','original_split','country','latitude','longitude','captured_at','capture_group','surface_type','original_label','proposed_quality_label','verified_quality_label','label_status','reviewer','review_date','permission_status','license','attribution','source_url','width','height','sha256','dhash','duplicate_group','read_status','eligible','exclusion_reasons','source_metadata']
PROPOSAL={'excellent':'Good','good':'Good','intermediate':'Moderate','bad':'Poor','very_bad':'Poor'}

def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path, rows, fields):
 with path.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def inspect(item):
 row,blob=item
 try:
  data=blob() if callable(blob) else Path(blob).read_bytes()
  row['sha256']=hashlib.sha256(data).hexdigest()
  with Image.open(io.BytesIO(data)) as im:
   im.load(); row['width'],row['height']=im.size
   img=ImageOps.exif_transpose(im).convert('L').resize((9,8)); px=list(img.getdata()); val=0
   for y in range(8):
    for x in range(8):val=(val<<1)|(px[y*9+x]>px[y*9+x+1])
   row['dhash']=f'{val:016x}'
  row['read_status']='readable'
 except Exception as e:
  row['read_status']='unreadable';row['read_error']=str(e);row['sha256']='';row['dhash']=''
 return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--limit-per-source',type=int);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 base=Path(a.dataset_root).resolve();out=Path(a.output)
 if not a.dry_run and out.exists():p.error('Output exists; choose a new output directory to avoid overwriting')
 tasks=[]; problems=[]; source_counts={}; rdd=base/'RDD_SPLIT'; cam=base/'RoadSense_Cambodian'; street=base/'streetSurfaceVis'
 for x in (rdd,cam,street):
  if not x.is_dir():p.error('Missing directory: '+str(x))
 paths=sorted(x for x in rdd.rglob('*') if x.is_file() and x.suffix.lower() in EXT and x.parent.name=='images')
 source_counts['RDD_SPLIT']=len(paths)
 for path in paths[:a.limit_per_source]:
  split=path.parent.parent.name;ann=path.parent.parent/'labels'/(path.stem+'.txt'); status='missing'
  if ann.exists():
   try:
    lines=ann.read_text(encoding='utf-8-sig').splitlines(); cnt=collections.Counter()
    for line in lines:
     f=line.split()
     if len(f)!=5:raise ValueError('expected five fields')
     cls=int(f[0]); values=list(map(float,f[1:]))
     if not all(0<=v<=1 for v in values):raise ValueError('coordinate outside [0,1]')
     cnt['class_'+str(cls)]+=1
    status='empty' if not cnt else 'annotated'
   except (ValueError,OSError) as e:status='invalid';problems.append(('RDD_SPLIT',str(ann),'invalid annotation: '+str(e)))
  else:problems.append(('RDD_SPLIT',str(ann),'missing annotation'))
  raw={'annotation_status':status,'damage_counts':dict(cnt) if status in ('empty','annotated') else {},'source_path':str(path)}
  row=dict(image_id='RDD:'+split+':'+path.stem,source='RDD_SPLIT',original_id=path.stem,image_path=str(path),annotation_path=str(ann) if ann.exists() else '',original_split=split,country=path.stem.rsplit('_',1)[0],original_label=json.dumps(raw['damage_counts'],sort_keys=True),label_status='unreviewed',permission_status='unknown',source_metadata=json.dumps(raw,sort_keys=True))
  tasks.append((row,path))
 data=read(street/'streetSurfaceVis_v1_0.csv'); files=sorted(x for x in (street/'s_1024').rglob('*') if x.is_file() and x.suffix.lower() in EXT); source_counts['streetSurfaceVis']=len(files)
 byid=collections.defaultdict(list)
 for r in data:byid[r['mapillary_image_id']].append(r)
 fileids={x.stem for x in files}
 for id_,rs in byid.items():
  if len(rs)!=1:problems.append(('streetSurfaceVis',id_,'duplicate CSV identifier'))
  if id_ not in fileids:problems.append(('streetSurfaceVis',id_,'CSV row missing image'))
 for path in files[:a.limit_per_source]:
  matches=byid.get(path.stem,[]); r=matches[0] if len(matches)==1 else {}
  if not r:problems.append(('streetSurfaceVis',str(path),'missing or ambiguous CSV record'))
  flag=r.get('train',''); split={'True':'train','False':'non_train'}.get(flag,'unknown')
  row=dict(image_id='SSV:'+path.stem,source='streetSurfaceVis',original_id=path.stem,image_path=str(path),annotation_path=str(street/'streetSurfaceVis_v1_0.csv'),original_split=split,latitude=r.get('latitude',''),longitude=r.get('longitude',''),captured_at=r.get('captured_at',''),surface_type=r.get('surface_type',''),original_label=r.get('surface_quality',''),proposed_quality_label=PROPOSAL.get(r.get('surface_quality',''),''),label_status='original five-level; conversion unapproved',permission_status='unknown',attribution=r.get('user_name',''),source_metadata=json.dumps(r,ensure_ascii=False,sort_keys=True))
  tasks.append((row,path))
 manifest=read(cam/'dataset_manifest.csv'); zpath=cam/'cambodia_road.zip'
 with zipfile.ZipFile(zpath) as z:
  names=[n for n in z.namelist() if not n.endswith('/') and Path(n).suffix.lower() in EXT and not n.startswith('__MACOSX/')]; source_counts['RoadSense_Cambodian']=len(names)
  bypath=collections.defaultdict(list)
  for r in manifest:bypath[r['path_in_archive']].append(r)
  for n in names[:a.limit_per_source]:
   matches=bypath.get(n,[]);r=matches[0] if len(matches)==1 else {}
   if not r:problems.append(('RoadSense_Cambodian',n,'missing or ambiguous prior manifest'))
   verified=r.get('review_label','') if r.get('review_status','').lower()=='reviewed' else ''
   row=dict(image_id='CAM:'+r.get('image_id',n),source='RoadSense_Cambodian',original_id=r.get('image_id',''),image_path=str(zpath),archive_member=n,original_split=r.get('split',''),country=r.get('country',''),capture_group=r.get('group_id',''),original_label=r.get('review_label',''),verified_quality_label=verified,label_status=r.get('review_status','unknown'),reviewer=r.get('reviewer',''),review_date=r.get('review_date',''),permission_status=r.get('training_permission','unknown'),license=r.get('license_permission_status',''),attribution=r.get('author',''),source_url=r.get('url',''),source_metadata=json.dumps(r,ensure_ascii=False,sort_keys=True))
   # Read under an independent ZipFile handle in each worker.
   def load(member=n):
    with zipfile.ZipFile(zpath) as archive:return archive.read(member)
   tasks.append((row,load))
  for n in bypath.keys()-set(names):problems.append(('RoadSense_Cambodian',n,'manifest original missing from archive'))
 with ThreadPoolExecutor(max_workers=a.workers) as pool: rows=list(pool.map(inspect,tasks))
 exact,edges,groups=group_duplicates(rows)
 comps=collections.defaultdict(list)
 for r in rows:
  if r.get('duplicate_group'):comps[r['duplicate_group']].append(r)
 for r in rows:
  reasons=[]
  if r['read_status']!='readable':reasons.append('unreadable');problems.append((r['source'],r['image_path'],r.get('read_error','unreadable')))
  if r.get('verified_quality_label') not in ('Good','Moderate','Poor'):reasons.append('no_verified_three_class_label')
  if r.get('permission_status','').lower() not in ('confirmed','granted','approved'):reasons.append('training_permission_unconfirmed')
  if r.get('duplicate_group'):reasons.append('duplicate_candidate_needs_review')
  r['eligible']=not reasons;r['exclusion_reasons']='|'.join(reasons)
 summary={'source_image_counts':source_counts,'processed_counts':dict(collections.Counter(r['source'] for r in rows)),'read_status':dict(collections.Counter(r['read_status'] for r in rows)),'source_quality_labels':{s:dict(collections.Counter(r.get('original_label','') for r in rows if r['source']==s)) for s in ('streetSurfaceVis','RoadSense_Cambodian')},'verified_labels':{s:dict(collections.Counter(r.get('verified_quality_label') or 'unverified' for r in rows if r['source']==s)) for s in source_counts},'eligible_by_source':dict(collections.Counter(r['source'] for r in rows if r['eligible'])),'exact_duplicate_extra_images':exact,'near_duplicate_components':groups,'images_in_duplicate_components':sum(map(len,comps.values())),'cross_source_components':sum(len({r['source'] for r in c})>1 for c in comps.values()),'original_split_crossing_components':sum(len({r['original_split'] for r in c if r['original_split'] in ('train','val','test','non_train')})>1 for c in comps.values()),'issue_count':len(problems),'is_sample':bool(a.limit_per_source)}
 print(json.dumps(summary,indent=2))
 if a.dry_run:return
 out.mkdir(parents=True);write(out/'unified_metadata.csv',rows,FIELDS);write(out/'training_ready.csv',[r for r in rows if r['eligible']],FIELDS);write(out/'issues.csv',[dict(source=s,path=path,issue=issue) for s,path,issue in problems],['source','path','issue']);write(out/'duplicate_candidates.csv',[dict(duplicate_group=k,image_id=r['image_id'],source=r['source'],original_split=r['original_split'],image_path=r['image_path'],archive_member=r.get('archive_member',''),sha256=r['sha256'],dhash=r['dhash']) for k,g in comps.items() for r in g],['duplicate_group','image_id','source','original_split','image_path','archive_member','sha256','dhash']);(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
if __name__=='__main__':main()
