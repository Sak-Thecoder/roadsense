"""Report each StreetSurfaceVis test image's nearest training coordinate."""
import csv,math,argparse
from scipy.spatial import cKDTree
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
with open(a.input,newline='',encoding='utf8') as f:rows=[r for r in csv.DictReader(f) if r['source']=='streetSurfaceVis']
train=[r for r in rows if r['original_split']=='train'];test=[r for r in rows if r['original_split']=='test']
def xyz(r):
 lat,lon=map(math.radians,(float(r['latitude']),float(r['longitude'])));return [math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)]
ds,indices=cKDTree([xyz(r) for r in train]).query([xyz(r) for r in test]);distances=[12742000*math.asin(min(1,float(d)/2)) for d in ds]
with open(a.output,'x',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=['test_id','nearest_train_id','distance_m']);w.writeheader();w.writerows({'test_id':r['original_id'],'nearest_train_id':train[int(i)]['original_id'],'distance_m':round(d,1)} for r,i,d in zip(test,indices,distances))
print('Minimum distance (m):',round(min(distances),1),'test images within 1 km:',sum(d<1000 for d in distances))
