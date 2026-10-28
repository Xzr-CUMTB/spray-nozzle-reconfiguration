import sys
sys.dont_write_bytecode=True
from functools import lru_cache
from common import ROOT,load
import frozen_model as frozen
from frozen_geometry import decode,feasible as base_feasible,repair,LOWER,UPPER
from geometry_gate import engineering_feasible
import numpy as np
BASELINE=np.array(load('layouts/C001_FROZEN_COPY.json')['candidate']['particle'])
def scene(name):return load(f'scenes/{name}.json')
@lru_cache(maxsize=10)
def free_volume(name,h):
 saved=frozen.SCENE
 try:
  frozen.SCENE=scene(name);return frozen.volumes(h,name!='EMPTY')
 finally:frozen.SCENE=saved
def local_bounds(subset):
 lo=BASELINE.copy();hi=BASELINE.copy()
 for i in subset:
  j=5*i;width=np.array([0.,.15,.15,30.,25.]);lo[j:j+5]=np.maximum(LOWER[j:j+5],BASELINE[j:j+5]-width);hi[j:j+5]=np.minimum(UPPER[j:j+5],BASELINE[j:j+5]+width);lo[j]=LOWER[j];hi[j]=UPPER[j]
 return lo,hi
def in_local(p,subset):
 lo,hi=local_bounds(subset);active=np.zeros(30,bool)
 for i in subset:active[5*i:5*i+5]=True
 return bool(np.asarray(p)[~active].tobytes()==BASELINE[~active].tobytes() and np.all(p>=lo) and np.all(p<=hi))
def global_draw(rng,name):
 p=repair(rng.uniform(LOWER,UPPER),rng)
 if not engineering_feasible(p,scene(name)):raise RuntimeError('Global initial draw violated scene; no extra repair change')
 return p
def local_draw(rng,subset,name):
 lo,hi=local_bounds(subset)
 for _ in range(10000):
  p=rng.uniform(lo,hi)
  for i in subset:p[5*i]=np.floor(p[5*i])+.5
  if engineering_feasible(p,scene(name)) and in_local(p,subset):return p
 raise RuntimeError('Local feasible initialization exhausted')
