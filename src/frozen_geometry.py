import numpy as np
import json
from pathlib import Path
from common import ROOT
P=json.loads((ROOT/'configs/NUMERICAL_PROTOCOL_FROZEN_COPY.json').read_text('utf-8'))
M=P['mounting']
LOWER=np.tile([0.,0.,0.,-180.,-90.],6);UPPER=np.tile([2.999999,1.,1.,180.,90.],6)
def decode(particle):
 rows=[]
 for i,(s,u,v,yaw,pitch) in enumerate(np.asarray(particle).reshape(-1,5)):
  si=min(int(s),2);x=M['axial_range'][0]+u*(M['axial_range'][1]-M['axial_range'][0])
  if si<2:y=M['left_y'] if si==0 else M['right_y'];z=M['side_z_range'][0]+v*(M['side_z_range'][1]-M['side_z_range'][0]);yaw+=(20. if si==0 else -20.)
  else:y=M['roof_y_range'][0]+v*(M['roof_y_range'][1]-M['roof_y_range'][0]);z=M['roof_z']
  rows.append(dict(nozzle=i+1,surface=['left_wall','right_wall','roof'][si],position=[float(x),float(y),float(z)],yaw_deg=float(yaw),pitch_deg=float(pitch)))
 return rows

def feasible(p):
 p=np.asarray(p)
 if p.shape!=(30,) or not np.isfinite(p).all() or np.any(p<LOWER) or np.any(p>UPPER):return False
 xyz=np.array([n['position'] for n in decode(p)])
 return all(not np.any(np.linalg.norm(xyz[i+1:]-xyz[i],axis=1)<M['min_spacing']) for i in range(6))

def repair(p,rng):
 q=np.clip(p,LOWER,UPPER).reshape(6,5).copy()
 for i in range(6):
  for attempt in range(200):
   rows=decode(q.ravel());xyz=np.array([n['position'] for n in rows]);ok=not np.any(np.linalg.norm(xyz[:i]-xyz[i],axis=1)<M['min_spacing'])
   if ok:break
   q[i,:3]=[rng.uniform(0.,2.999999),rng.random(),rng.random()]
  else:raise RuntimeError('Repair exhausted')
 return q.ravel()
