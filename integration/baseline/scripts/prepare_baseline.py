"""Create a stratified mapping review gallery and provisional manifests."""
import csv, json, random, html, sys, collections, math
from pathlib import Path
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
OUT=ROOT/'integration/baseline'; DATA=ROOT/'dataset'; OLD=ROOT/'integration/audit_2026-11-15'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows,fields):
 with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
rows=read(OLD/'unified_metadata.csv'); street=[r for r in rows if r['source']=='streetSurfaceVis']; camb=[r for r in rows if r['source']=='RoadSense_Cambodian']; original=read(DATA/'RoadSense_Cambodian/dataset_manifest.csv')
# No inferred Good/Moderate/Poor label becomes verified.
for r in street:
 r['original_split']='train' if json.loads(r['source_metadata']).get('train')=='True' else 'test'
 r['country']='Germany (dataset authors report bounding-box selection)';r['license']='CC BY-SA 4.0 (dataset-level; official source and file checksums matched)';r['permission_status']='dataset_license_evidence; intended use review pending'
fields=list(rows[0]);
for path in [OUT/'manifests/street_provisional.csv', OUT/'manifests/cambodian_review_working.csv', OUT/'manifests/unified_current.csv', OUT/'gallery/index.html']:
 if path.exists(): raise SystemExit(f'Refusing to overwrite existing output: {path}')
write(OUT/'manifests/street_provisional.csv',street,fields)
write(OUT/'manifests/cambodian_review_working.csv',original,list(original[0]))
write(OUT/'manifests/unified_current.csv',rows,fields)
# Balanced strata, not prevalence-weighted; seeded random image selection avoids repeated perceptual components.
rng=random.Random(42); strata=collections.defaultdict(list)
for r in street:strata[(r['original_label'],r['surface_type'],r['original_split'])].append(r)
selected=[]; seen=set()
for key in sorted(strata):
 candidates=strata[key];rng.shuffle(candidates);count=0
 for r in candidates:
  group=r['duplicate_group'] or r['image_id']
  if group in seen:continue
  selected.append(r);seen.add(group);count+=1
  if count==3:break
folder=OUT/'gallery';thumbs=folder/'thumbs';thumbs.mkdir(exist_ok=True)
cards=[]
for r in selected:
 src=Path(r['image_path']);name=r['original_id']+'.jpg'
 with Image.open(src) as im:
  im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((480,300));im.save(thumbs/name,quality=80)
 caption=' | '.join([r['original_id'],r['surface_type'],r['original_label'],'proposed '+r['proposed_quality_label'],r['original_split']])
 cards.append('<figure><img loading="lazy" src="thumbs/'+html.escape(name)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
(folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>StreetSurfaceVis mapping review</title><style>body{font:16px system-ui;margin:2rem}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1rem}figure{margin:0;border:1px solid #ccc;padding:.5rem}img{width:100%;height:190px;object-fit:contain}figcaption{font-size:.86rem}</style><h1>StreetSurfaceVis mapping review</h1><p>Seed 42; up to three images per original rating, surface type and source split, excluding repeated detected similarity components. Stratified for coverage, not prevalence-representative. Proposed labels are not approved; compare with the original guide.</p><main>'+''.join(cards)+'</main>',encoding='utf-8')
write(folder/'selection.csv',selected,fields)
# Direct (not transitive) cross-split perceptual candidates, with distance and geo distance.
train=[r for r in street if r['original_split']=='train']; test=[r for r in street if r['original_split']=='test']; buckets=collections.defaultdict(list)
for r in train:
 v=int(r['dhash'],16)
 for block in range(5):
  bits=12 if block==4 else 13;buckets[(block,(v>>(13*block))&((1<<bits)-1))].append(r)
pairs=[]
def distance(a,b):
 try:
  lat1,lat2=map(math.radians,[float(a['latitude']),float(b['latitude'])]);dlon=math.radians(float(a['longitude'])-float(b['longitude']));dlat=lat1-lat2
  q=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
  return round(12742000*math.asin(min(1,math.sqrt(q))),1)
 except (ValueError,KeyError):return ''
for t in test:
 v=int(t['dhash'],16);cands={}
 for block in range(5):
  bits=12 if block==4 else 13
  for r in buckets[(block,(v>>(13*block))&((1<<bits)-1))]:cands[r['image_id']]=r
 for r in cands.values():
  h=(v^int(r['dhash'],16)).bit_count()
  if h<=4:pairs.append({'train_id':r['original_id'],'test_id':t['original_id'],'dhash_distance':h,'exact_sha256':r['sha256']==t['sha256'],'distance_m':distance(r,t),'review_decision':''})
write(OUT/'reports/street_cross_split_candidates.csv',pairs,['train_id','test_id','dhash_distance','exact_sha256','distance_m','review_decision'])
print(json.dumps({'gallery_images':len(selected),'gallery_strata':len(set((r['original_label'],r['surface_type'],r['original_split']) for r in selected)),'direct_cross_split_candidates':len(pairs),'cross_split_exact':sum(x['exact_sha256'] for x in pairs),'within_100m_candidates':sum(isinstance(x['distance_m'],float) and x['distance_m']<=100 for x in pairs)},indent=2))
