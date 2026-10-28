# -*- coding: utf-8 -*-
"""Spray sampling, nozzle decoding, and exact free-air voxel volumes."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtri
from event_kernel import integrate_event
from common import ROOT
P=json.loads((ROOT/'configs/NUMERICAL_PROTOCOL_FROZEN_COPY.json').read_text('utf-8'))
SCENE={'domain':P['domain_m'],'target':P['target_m'],'mounting':P['mounting'],'solid_geometry':[]}
NOZZLE=P['physical_inputs'];FIT=P['diameter_distribution'];M=SCENE['mounting']
FIDELITIES={'P0':(.10,200),'P4':(.04,800)}
LOWER=np.tile([0.,0.,0.,-180.,-90.],6);UPPER=np.tile([2.999999,1.,1.,180.,90.],6)


def shape_for(h):return tuple(np.ceil((np.array(SCENE['domain'][1])-SCENE['domain'][0])/h-1.e-10).astype(int))

def volumes(h,obstacles):
 lo,hi=np.array(SCENE['target']);origin=np.array(SCENE['domain'][0]);shape=shape_for(h)
 if not obstacles:
  ds=[np.maximum(0.,np.minimum(origin[d]+(np.arange(shape[d])+1)*h,hi[d])-np.maximum(origin[d]+np.arange(shape[d])*h,lo[d])) for d in range(3)]
  return ds[0][:,None,None]*ds[1][None,:,None]*ds[2][None,None,:]
 boxes=np.array([[b['lower'],b['upper']] for b in SCENE['solid_geometry']]);edges=[np.unique(np.clip(np.r_[origin[d]+np.arange(shape[d]+1)*h,lo[d],hi[d],boxes[:,:,d].ravel()],lo[d],hi[d])) for d in range(3)];mids=[(e[1:]+e[:-1])/2 for e in edges];width=[np.diff(e) for e in edges]
 iy=np.floor((mids[1]-origin[1])/h).astype(int);iz=np.floor((mids[2]-origin[2])/h).astype(int);linear=(iy[:,None]*shape[2]+iz[None,:]).ravel();yy,zz=np.meshgrid(mids[1],mids[2],indexing='ij');area=width[1][:,None]*width[2][None,:];out=np.zeros(shape)
 for x,dx in zip(mids[0],width[0]):
  inside=np.zeros(yy.shape,bool)
  for b in boxes:
   if b[0,0]<=x<=b[1,0]:inside|=(yy>=b[0,1])&(yy<=b[1,1])&(zz>=b[0,2])&(zz<=b[1,2])
  ix=int(np.floor((x-origin[0])/h));out[ix]+=np.bincount(linear,weights=(np.where(inside,0.,area)*dx).ravel(),minlength=shape[1]*shape[2]).reshape(shape[1:])
 return out

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

def initial(nozzle,n,seed):
 rng=np.random.default_rng(seed);u=np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(3)])
 d=np.exp(FIT['mu_ln_m']+FIT['sigma_ln']*ndtri(u[:,0]));yaw=np.deg2rad(nozzle['yaw_deg']);pitch=np.deg2rad(nozzle['pitch_deg']);axis=np.array([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
 ref=np.array([0.,0.,1.]) if abs(axis[2])<.95 else np.array([0.,1.,0.]);b1=np.cross(axis,ref);b1/=np.linalg.norm(b1);b2=np.cross(axis,b1);ct=1-u[:,1]*(1-np.cos(np.deg2rad(NOZZLE['cone_angle_deg']/2)));st=np.sqrt(1-ct*ct);az=2*np.pi*u[:,2]
 v=NOZZLE['initial_speed_mps']*(ct[:,None]*axis+st[:,None]*(np.cos(az)[:,None]*b1+np.sin(az)[:,None]*b2))
 # Reference liquid volume for a 1-second injection-equivalent pulse. This
 # normalization duration is NOT the trajectory integration time.
 w=np.full(n,(NOZZLE['flow_rate_L_min']*1.e-3/60.)/n)
 return np.array(nozzle['position'],float),v,d,w

def volume_fractions(states,weights):
 den=weights.sum();return {name:float(weights[states==code].sum()/den) for name,code in [('active',0),('exited',1),('collision',2)]}
