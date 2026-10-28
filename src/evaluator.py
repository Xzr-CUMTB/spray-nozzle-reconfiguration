"""Unchanged trajectory/free-volume functions with scene translation and exact component reuse."""
import sys
sys.dont_write_bytecode=True
import os,time,hashlib
if os.name == 'nt':
 import msvcrt
else:
 import fcntl
import numpy as np
from common import *
from scene_model import scene,free_volume,BASELINE,engineering_feasible
from frozen_model import decode,initial,volume_fractions,FIDELITIES
from event_kernel import integrate_event
def digest(p):return hashlib.sha256(np.asarray(p,dtype=np.float64).tobytes()).hexdigest()
class CacheLock:
 def __init__(self,path):self.path=path
 def __enter__(self):
  self.f=open(self.path,'a+b');self.f.seek(0,2)
  if self.f.tell()==0:self.f.write(b'0');self.f.flush()
  while True:
   try:
    self.f.seek(0)
    if os.name == 'nt':msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
    else:fcntl.flock(self.f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    return self
   except OSError:
    if elapsed()>load('configs/STAGE_BUDGET.json')['stop_trajectory_s']:raise RuntimeError('Wall stop during component cache lock')
    time.sleep(.03)
 def __exit__(self,*args):
  self.f.seek(0)
  if os.name == 'nt':msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1)
  else:fcntl.flock(self.f.fileno(),fcntl.LOCK_UN)
  self.f.close()
def trajectory(nozzle,name,fidelity,seed,shared_mask=None):
 sc=scene(name);h,n=FIDELITIES[fidelity];free=free_volume(name,h);mask=np.zeros(free.shape,bool) if shared_mask is None else shared_mask;p,v,d,w=initial(nozzle,n,seed);a=np.array([[b['lower'],b['upper']] for b in sc['solid_geometry']],float).reshape(-1,2,3)
 start=time.perf_counter()
 state,t,duration,steps,mindt,maxdt,jump,_,_,v_end,p_end=integrate_event(p,v,d,w,np.array([1.,0.,0.]),np.array(sc['domain'][0]),np.array(sc['domain'][1]),h,.002,a,mask,12.,.2,False,True)
 if not np.isfinite(v_end).all() or not np.isfinite(p_end).all():raise RuntimeError('NONFINITE_TRAJECTORY')
 stats=dict(steps=int(steps),elapsed_s=float(duration),min_dt_s=float(mindt),max_dt_s=float(maxdt),max_step_voxels=float(jump),wall_s=time.perf_counter()-start)
 return mask,state,w,stats
def component(nozzle,row,name,fidelity,seed,cache_enabled):
 eligible=any(np.asarray(row,dtype=np.float64).tobytes()==base.tobytes() for base in BASELINE.reshape(6,5))
 if not(cache_enabled and eligible):return (*trajectory(nozzle,name,fidelity,seed),False)
 key=hashlib.sha256((sha(ROOT/f'scenes/{name}.json')+f'|{fidelity}|{seed}|'+digest(row)).encode()).hexdigest();folder=ROOT/'component_cache';folder.mkdir(exist_ok=True);path=folder/(key+'.npz')
 with CacheLock(folder/(key+'.lock')):
  if path.exists():
   with np.load(path,allow_pickle=False) as z:return z['mask'],z['state'],z['weights'],json.loads(str(z['stats'])),True
  mask,state,w,stats=trajectory(nozzle,name,fidelity,seed);tmp=folder/(key+'.tmp.npz');np.savez_compressed(tmp,mask=mask,state=state,weights=w,stats=json.dumps(stats));tmp.replace(path);return mask,state,w,stats,False
def evaluate(label,p,name,fidelity,seed,phase,save_mask=False,cache_enabled=True):
 path=ROOT/'results'/f'{label}.json';pd=digest(p)
 if path.exists():
  out=load(f'results/{label}.json');assert out['particle_sha256']==pd and out['scene']==name and out['seed']==seed and out['fidelity']==fidelity and out['cache_enabled']==cache_enabled;return out
 sc=scene(name);assert engineering_feasible(p,sc),'Infeasible layout before evaluation'
 ident=reserve(label,phase,name,fidelity,seed,pd);start=time.perf_counter();hits=0
 try:
  h,n=FIDELITIES[fidelity];free=free_volume(name,h);covered=np.zeros(free.shape,bool);states=[];weights=[];stats=[]
  for nozzle,row in zip(decode(p),np.asarray(p).reshape(6,5)):
   if elapsed()>load('configs/STAGE_BUDGET.json')['stop_trajectory_s']:raise RuntimeError('TRAJECTORY_WALL_STOP')
   if cache_enabled:mask,state,w,st,hit=component(nozzle,row,name,fidelity,seed,True)
   else:mask,state,w,st=trajectory(nozzle,name,fidelity,seed,shared_mask=covered);hit=False
   covered|=mask;states.append(state);weights.append(w);stats.append(st);hits+=int(hit)
  fractions=volume_fractions(np.concatenate(states),np.concatenate(weights));status='EVENT_COMPLETE' if fractions['active']==0 else ('VOLUME_TOLERANCE_COMPLETE' if fractions['active']<.005 else 'HORIZON_INCOMPLETE')
  if status=='HORIZON_INCOMPLETE':raise RuntimeError(status)
  out=dict(label=label,direct_id=ident,scene=name,fidelity=fidelity,phase=phase,seed=seed,particle_sha256=pd,coverage=float(free[covered].sum()/free.sum()),visited_volume_m3=float(free[covered].sum()),free_target_volume_m3=float(free.sum()),status=status,fractions=fractions,stats=stats,wall_s=time.perf_counter()-start,cache_enabled=cache_enabled,baseline_component_hits=hits,new_nozzle_trajectory_calls=6-hits,visited_sha256=hashlib.sha256(covered.tobytes()).hexdigest())
  if save_mask:np.savez_compressed(ROOT/'results'/f'{label}_mask.npz',covered=covered,free_volume_m3=free)
  dump(f'results/{label}.json',out);finish(ident,out['wall_s']);return out
 except Exception as exc:finish(ident,time.perf_counter()-start,repr(exc));raise
