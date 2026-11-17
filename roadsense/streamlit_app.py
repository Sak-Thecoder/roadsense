"""Human review of a pilot CSV; decisions are stored separately from metadata."""
import csv
import os
import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title='RoadSense review', layout='wide')
st.title('RoadSense road-quality review')
st.caption('Review the road surface visually. Damage annotations and empty annotations do not determine quality.')
pilot_path = Path(st.sidebar.text_input('Pilot CSV', 'work/full/pilot.csv'))
labels_path = Path(st.sidebar.text_input('Label progress CSV', 'work/full/pilot_labels.csv'))
if not pilot_path.is_file():
    st.info('Enter the path to an existing pilot CSV.'); st.stop()
with pilot_path.open(newline='', encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
if not rows: st.warning('Pilot is empty.'); st.stop()
labels={}
if labels_path.exists():
    with labels_path.open(newline='',encoding='utf-8-sig') as f: labels={r['image_id']:r['quality_label'] for r in csv.DictReader(f)}
valid={'Good','Moderate','Poor','Uncertain','Reject'}
counts={v:sum(x==v for x in labels.values()) for v in valid}
st.sidebar.write(f"Reviewed: {sum(r['image_id'] in labels for r in rows)} / {len(rows)}")
st.sidebar.write(counts)
if 'idx' not in st.session_state: st.session_state.idx=next((i for i,r in enumerate(rows) if r['image_id'] not in labels),0)
st.session_state.idx=min(st.session_state.idx,len(rows)-1)
a,b,c=st.columns([1,1,4])
if a.button('Previous'): st.session_state.idx=max(0,st.session_state.idx-1)
if b.button('Next'): st.session_state.idx=min(len(rows)-1,st.session_state.idx+1)
row=rows[st.session_state.idx]
st.subheader(f"{st.session_state.idx+1} / {len(rows)} — {row['original_id']}")
st.write(f"Source: {row['source']} · Country/prefix: {row['country']} · Current decision: {labels.get(row['image_id'], 'Not reviewed')}")
try:
    with Image.open(row['image_path']) as im: st.image(ImageOps.exif_transpose(im), use_container_width=True)
except OSError as e: st.error(f'Cannot open image: {e}')
st.caption('Recorded damage categories: '+(row.get('damage_types') or 'none recorded'))
cols=st.columns(5)
for col,label in zip(cols,('Good','Moderate','Poor','Uncertain','Reject')):
    if col.button(label, key=label, use_container_width=True):
        labels[row['image_id']]=label
        labels_path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(dir=labels_path.parent,suffix='.csv')
        try:
            with os.fdopen(fd,'w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=['image_id','quality_label']); w.writeheader()
                w.writerows({'image_id':k,'quality_label':v} for k,v in sorted(labels.items()))
            os.replace(tmp,labels_path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        st.session_state.idx=min(st.session_state.idx+1,len(rows)-1); st.rerun()
