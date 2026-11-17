"""Audit an extracted RDD dataset and create a review pilot without changing originals."""
import argparse
import csv
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
FIELDS = ['image_id','image_path','source','country','original_id','original_split','annotation_path','annotation_format','annotation_status','damage_types','damage_counts','width','height','sha256','dhash','duplicate_group','scene_group','quality_label','license','attribution','source_url']

def csv_write(path, rows, fields):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def discover(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXT and '__MACOSX' not in p.parts and not p.name.startswith('._'))

def process(item):
    root, p = item
    rel = p.relative_to(root); parts = rel.parts
    split = next((x for x in parts if x in ('train','val','test')), '')
    country = re.sub(r'_\d+$', '', p.stem)
    candidates = [p.with_suffix('.xml')]
    if p.parent.name == 'images':
        candidates += [p.parent.parent / 'annotations' / (p.stem + '.xml'), p.parent.parent / 'labels' / (p.stem + '.txt')]
    ann = next((x for x in candidates if x.exists()), None)
    counts = Counter(); errors = []; status = 'missing'; fmt = ''
    if ann:
        fmt = ann.suffix.lower()[1:]
        try:
            if fmt == 'xml':
                counts.update((x.text or '').strip() for x in ET.parse(ann).findall('.//object/name') if (x.text or '').strip())
            else:
                for line in ann.read_text(encoding='utf-8-sig').splitlines():
                    fields = line.split()
                    if len(fields) != 5: raise ValueError('expected five YOLO fields')
                    cls = int(fields[0]); values = [float(x) for x in fields[1:]]
                    if any(not 0 <= x <= 1 for x in values): raise ValueError('YOLO coordinate outside [0,1]')
                    counts['class_' + str(cls)] += 1
            status = 'annotated' if counts else 'empty'
        except (OSError, ValueError, ET.ParseError) as e:
            status = 'invalid'; errors.append({'path':str(ann),'issue':str(e)})
    else: errors.append({'path':str(p),'issue':'annotation missing'})
    sha = hashlib.sha256(); width = height = ''; dh = ''
    try:
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''): sha.update(block)
        with Image.open(p) as im:
            im.load(); width,height = im.size
            im = ImageOps.exif_transpose(im).convert('L').resize((9,8)); px = list(im.getdata()); val = 0
            for y in range(8):
                for x in range(8): val = (val << 1) | (px[y*9+x] > px[y*9+x+1])
            dh = f'{val:016x}'
    except (OSError, ValueError) as e:
        sha = None; errors.append({'path':str(p),'issue':'unreadable image: '+str(e)})
    row = dict(image_id=hashlib.sha256(str(rel).encode()).hexdigest()[:20], image_path=str(p), source='RDD2022', country=country, original_id=p.stem, original_split=split, annotation_path=str(ann) if ann else '', annotation_format=fmt, annotation_status=status, damage_types='|'.join(sorted(counts)), damage_counts=json.dumps(dict(counts),sort_keys=True), width=width,height=height,sha256=sha.hexdigest() if sha else '',dhash=dh,duplicate_group='',scene_group='',quality_label='',license='',attribution='',source_url='')
    return row, errors

def group_duplicates(rows, threshold=4):
    # Five disjoint bit blocks: two hashes <=4 bits apart share at least one block.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a != b: parent[b]=a
    exact = {}; unique = {}; exact_pairs = 0
    for i,r in enumerate(rows):
        sha=r['sha256']; dh=r['dhash']
        if sha:
            if sha in exact: union(i,exact[sha]); exact_pairs+=1
            else: exact[sha]=i
        if dh:
            if dh in unique: union(i,unique[dh])
            else: unique[dh]=i
    buckets=defaultdict(list); near_edges=0
    for h,i in unique.items():
        v=int(h,16); candidates=set()
        for block in range(5):
            shift=block*13; bits=12 if block==4 else 13; key=(block,(v>>shift)&((1<<bits)-1))
            candidates.update(buckets[key]); buckets[key].append((v,i))
        for other,j in candidates:
            if (v^other).bit_count() <= threshold:
                if find(i)!=find(j): union(i,j); near_edges+=1
    components=defaultdict(list)
    for i in range(len(rows)): components[find(i)].append(i)
    for members in components.values():
        if len(members)>1:
            group='dup_'+min(rows[i]['image_id'] for i in members)
            for i in members: rows[i]['duplicate_group']=group
    return exact_pairs, near_edges, sum(len(v)>1 for v in components.values())

def pilot(rows, size, seed):
    rng=random.Random(seed); eligible=[r for r in rows if r['sha256'] and r['annotation_status']!='invalid']
    rng.shuffle(eligible); strata=defaultdict(list)
    for r in eligible:
        types=r['damage_types'].split('|') if r['damage_types'] else ['no_recorded_boxes']
        for t in types: strata[(r['country'],t)].append(r)
    chosen=[]; used=set(); selected=set(); keys=sorted(strata)
    while len(chosen)<size and keys:
        nextkeys=[]
        for key in keys:
            bucket=strata[key]
            while bucket:
                r=bucket.pop(); group=r['duplicate_group'] or r['image_id']
                if group not in used and r['image_id'] not in selected:
                    chosen.append(r); used.add(group); selected.add(r['image_id']); break
            if bucket: nextkeys.append(key)
            if len(chosen)>=size: break
        keys=nextkeys
    return chosen

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out',required=True); ap.add_argument('--limit',type=int); ap.add_argument('--pilot-size',type=int,default=450); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
    root=Path(a.root).resolve(); out=Path(a.out)
    if not root.is_dir(): ap.error('root must be a directory')
    paths=discover(root); paths=paths[:a.limit] if a.limit else paths
    with ThreadPoolExecutor(max_workers=a.workers) as pool: results=list(pool.map(process,((root,p) for p in paths)))
    rows=[r for r,e in results]; errors=[x for r,e in results for x in e]
    exact,edges,groups=group_duplicates(rows); selected=pilot(rows,a.pilot_size,a.seed)
    countries=Counter(r['country'] for r in rows); status=Counter(r['annotation_status'] for r in rows); classes=Counter(); co=Counter()
    components=defaultdict(list)
    for r in rows:
        if r['duplicate_group']: components[r['duplicate_group']].append(r)
    cross_splits=sum(len({r['original_split'] for r in g if r['original_split']}) > 1 for g in components.values())
    country_annotations=Counter(r['country'] for r in rows if r['annotation_path'])
    for r in rows:
        classes.update(json.loads(r['damage_counts'])); types=sorted(filter(None,r['damage_types'].split('|')))
        for i,x in enumerate(types):
            for y in types[i+1:]: co[(x,y)]+=1
    summary={'images':len(rows),'countries':dict(countries),'annotation_status':dict(status),'annotation_formats':dict(Counter(r['annotation_format'] or 'missing' for r in rows)),'box_counts':dict(classes),'co_occurrence_images':{' + '.join(k):v for k,v in co.items()},'unreadable_images':sum('unreadable image:' in x['issue'] for x in errors),'exact_duplicate_extra_images':exact,'duplicate_components':groups,'duplicate_component_images':sum(map(len,components.values())),'cross_original_split_components':cross_splits,'annotation_files_by_country':dict(country_annotations),'near_union_edges':edges,'pilot_images':len(selected),'pilot_countries':dict(Counter(r['country'] for r in selected)),'sample_limit':a.limit,'seed':a.seed}
    print(json.dumps(summary,indent=2))
    if a.dry_run: return
    out.mkdir(parents=True,exist_ok=True); csv_write(out/'metadata.csv',rows,FIELDS); csv_write(out/'pilot.csv',selected,FIELDS); csv_write(out/'issues.csv',errors,['path','issue']); (out/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,5)); ax[0].barh(list(countries),list(countries.values())); ax[0].set_title('Images by filename prefix'); ax[1].bar(list(classes),list(classes.values())); ax[1].set_title('Bounding boxes by recorded class'); fig.tight_layout(); fig.savefig(out/'audit_charts.png',dpi=150); plt.close(fig)
    def table(d): return '\n'.join(f'| {k} | {v} |' for k,v in d.items()) or '| None | 0 |'
    report=f"""# Dataset audit\n\nRoot: `{root}`. Images: **{len(rows)}**. {'Sample only.' if a.limit else 'All discovered images.'}\n\n## Image filename prefixes\n\n| Prefix | Images |\n|---|---:|\n{table(countries)}\n\n## Annotation files by filename prefix\n\n| Prefix | Files |\n|---|---:|\n{table(country_annotations)}\n\n## Annotation status\n\n| Status | Images |\n|---|---:|\n{table(status)}\n\n## Bounding boxes\n\n| Recorded class | Boxes |\n|---|---:|\n{table(classes)}\n\n## Co-occurrences\n\nCounts are images containing both classes.\n\n| Pair | Images |\n|---|---:|\n{table(summary['co_occurrence_images'])}\n\n## Findings\n\n- Unreadable images: {summary['unreadable_images']}; missing and invalid annotations are detailed in issues.csv.\n- Exact duplicate extra images: {exact}; perceptual duplicate components: {groups}. Components use a 64-bit difference hash at distance <= 4 and can contain false positives and false negatives.\n- Duplicate components spanning existing splits: {cross_splits}; similarity candidates require manual review.\n- Existing splits and filename prefixes do not prove scene independence or geographic provenance. No sequence IDs were inferred.\n- Numeric YOLO IDs are preserved without class mapping. Empty annotations are not road-quality labels.\n- Pilot: {len(selected)} images, seed {a.seed}; one per detected duplicate component.\n\n![Audit charts](audit_charts.png)\n"""
    (out/'REPORT.md').write_text(report,encoding='utf-8')
if __name__=='__main__': main()
