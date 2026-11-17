"""Validate human decisions and make reproducible group-aware future splits."""
import argparse, csv, json, random
from collections import Counter, defaultdict
from pathlib import Path
from roadsense.audit import csv_write, FIELDS

def read(path):
    with open(path,newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def prepare(rows, labels, seed, minimum):
    if len({r['image_id'] for r in rows}) != len(rows): raise ValueError('Duplicate image IDs')
    decisions={}
    for r in labels:
        if r['image_id'] in decisions: raise ValueError('Duplicate label IDs')
        if r['quality_label'] not in {'Good','Moderate','Poor','Uncertain','Reject'}: raise ValueError('Invalid human label')
        decisions[r['image_id']]=r['quality_label']
    if set(decisions)-{r['image_id'] for r in rows}: raise ValueError('Labels contain IDs absent from metadata')
    parent=list(range(len(rows))); seen={}
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j): parent[find(i)]=find(j)
    for i,r in enumerate(rows):
        for col in ('duplicate_group','scene_group'):
            if r.get(col):
                key=(col,r[col])
                if key in seen: union(i,seen[key])
                else: seen[key]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[find(i)].append(r)
    items=list(groups.values()); random.Random(seed).shuffle(items)
    # Local holdout only when enough separate all-Cambodian components exist.
    local=[g for g in items if all(r.get('country','').lower() in ('cambodia','kh') for r in g) and any(decisions.get(r['image_id']) in ('Good','Moderate','Poor') for r in g)]
    hold=set()
    if len(local)>=10:
        for g in local[:max(1,round(len(local)*.2))]: hold.add(id(g))
    usable=lambda g: sum(decisions.get(r['image_id']) in ('Good','Moderate','Poor') for r in g)
    total=sum(usable(g) for g in items if id(g) not in hold); targets={'train':total*.7,'validation':total*.15,'test':total*.15}; counts=Counter(); output=[]
    for g in sorted(items,key=usable,reverse=True):
        if id(g) in hold: split='cambodia_holdout'
        else:
            split=max(targets,key=lambda k:(targets[k]-counts[k])/max(1,targets[k])); counts[split]+=usable(g)
        for r in g:
            label=decisions.get(r['image_id'],''); output.append(dict(r,quality_label=label,split=split,eligible=label in ('Good','Moderate','Poor')))
    distribution=Counter(decisions.values()); shortages={k:max(0,minimum-distribution[k]) for k in ('Good','Moderate','Poor')}
    return output, {'reviewed':len(decisions),'labels':dict(distribution),'shortages_to_minimum':shortages,'minimum_per_class':minimum,'eligible_split_counts':dict(counts),'cambodian_holdout_created':bool(hold),'missing_scene_groups':sum(not r.get('scene_group') for r in rows),'note':'Group and class balance are approximate; verify actual scene metadata and split distributions.'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--metadata',required=True); p.add_argument('--labels',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=42); p.add_argument('--minimum-per-class',type=int,default=50); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
    rows,summary=prepare(read(a.metadata),read(a.labels),a.seed,a.minimum_per_class)
    if not a.dry_run: csv_write(a.output,rows,FIELDS+['split','eligible'])
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
