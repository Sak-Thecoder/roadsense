"""Correct the saved Cambodian decisions without reopening or relabeling originals."""
import csv,datetime,json,sys
from pathlib import Path
import pandas as pd
import streamlit as st
BASE=Path(__file__).resolve().parents[1]
st.title('RoadSense Cambodian label corrections')
st.caption('Your confirmed labels are prefilled. Edit only what needs correction; review and training permission are separate.')
source=Path(st.sidebar.text_input('Review CSV',str(BASE/'manifests/owner_confirmed/cambodian_reviewed.csv')))
if not source.exists():st.error('Review CSV not found');st.stop()
with source.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
cols=['image_id','source','review_label','review_status','reviewer','review_date','group_id','training_permission','rights_notes','review_notes']
data=pd.DataFrame([{k:r.get(k,'') for k in cols} for r in rows])
st.write('Recorded labels:',data['review_label'].value_counts().to_dict())
st.info('Leave an unknown inspection date blank. CAM022 and CAM023 are identical; both records stay in the dataset.')
edited=st.data_editor(data,hide_index=True,num_rows='fixed',column_config={'image_id':st.column_config.TextColumn(disabled=True),'source':st.column_config.TextColumn(disabled=True),'review_label':st.column_config.SelectboxColumn(options=['Good','Moderate','Poor','Uncertain','Reject']),'review_status':st.column_config.SelectboxColumn(options=['reviewed','unreviewed']),'training_permission':st.column_config.SelectboxColumn(options=['confirmed','unconfirmed','denied'])},use_container_width=True)
if st.button('Save a new version'):
 values=edited.to_dict('records');changed={v['image_id']:v for v in values}
 for r in rows:
  if r['image_id'] in changed:
   for k in cols[2:]:r[k]=str(changed[r['image_id']].get(k,'') or '')
 out=BASE/'manifests'/'review_versions';out.mkdir(exist_ok=True);file=out/('review_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv')
 with file.open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 st.success(f'Saved {file}. Run the import command in the README to synchronize it with unified metadata.')
