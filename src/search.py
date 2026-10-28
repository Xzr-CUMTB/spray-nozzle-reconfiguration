import sys,os
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
import copy,time
from common import *
from scene_model import *
from evaluator import evaluate
def postcheck(q,previous,subset,name):
 reasons=[]
 if subset is not None:
  lo,hi=local_bounds(subset);active=np.zeros(30,bool)
  for i in subset:active[5*i:5*i+5]=True
  if q[~active].tobytes()!=BASELINE[~active].tobytes():reasons.append('frozen_parameter_changed')
  if np.any(q[active]<lo[active]) or np.any(q[active]>hi[active]):reasons.append('selected_radius_violation')
 if not engineering_feasible(q,scene(name)):reasons.append('scene_engineering_constraint')
 if reasons:q=previous.copy()
 assert engineering_feasible(q,scene(name))
 if subset is not None:assert in_local(q,subset)
 return q,reasons
class State:
 def __init__(self,job):
  self.job=job;self.rng=np.random.default_rng(job['optimizer_seed']);self.subset=None if job['phase']=='K6' else job['subset'];self.lo,self.hi=(LOWER.copy(),UPPER.copy()) if self.subset is None else local_bounds(self.subset)
  init_path=f"layouts/{job['id']}_initial.json"
  if (ROOT/init_path).exists():
   ini=load(init_path);check_frozen(init_path);assert ini['job']==job;self.positions=np.array(ini['positions']);self.rng.bit_generator.state=ini['post_initialization_rng_state']
  else:
   self.positions=np.array([BASELINE.copy()]+[(global_draw(self.rng,job['scene']) if self.subset is None else local_draw(self.rng,self.subset,job['scene'])) for _ in range(job['particles']-1)])
   freeze(init_path,dict(job=job,positions=self.positions.tolist(),post_initialization_rng_state=self.rng.bit_generator.state,C001_particle0_exact=True))
  self.velocity=np.zeros_like(self.positions);self.personal=self.positions.copy();self.scores=np.full(job['particles'],-np.inf);self.best=None;self.best_score=-np.inf;self.generation=-1;self.repairs=0;self.rollback_count=0;self.reasons={k:0 for k in ['frozen_parameter_changed','selected_radius_violation','scene_engineering_constraint']};self.history=[];self.archive=[]
 def propose(self,generation):
  if generation==0:return self.positions.copy()
  inertia=.9+(.4-.9)*generation/self.job['updates'];particles,dimensions=self.positions.shape
  rp=self.rng.random((particles,dimensions));rg=self.rng.random((particles,dimensions));pd=self.personal-self.positions;gd=self.best-self.positions;yaw=np.arange(3,dimensions,5);surface_ids=np.arange(0,dimensions,5)
  pd[:,yaw]=(pd[:,yaw]+180)%360-180;gd[:,yaw]=(gd[:,yaw]+180)%360-180
  self.velocity=inertia*self.velocity+1.55*rp*pd+1.75*rg*gd
  self.velocity=np.clip(self.velocity,-.2*(self.hi-self.lo),.2*(self.hi-self.lo))
  proposed=self.positions+self.velocity;proposed[:,yaw]=(proposed[:,yaw]+180)%360-180
  draw=self.rng.random((particles,6));surface=np.floor(self.positions[:,surface_ids]);surface=np.where(draw<.35,np.floor(self.personal[:,surface_ids]),surface);surface=np.where((draw>=.35)&(draw<.70),np.floor(self.best[surface_ids]),surface);surface=np.where(draw>=.95,self.rng.integers(0,3,surface.shape),surface);proposed[:,surface_ids]=surface+.5;self.velocity[:,surface_ids]=0.
  clipped=np.clip(proposed,self.lo,self.hi);self.velocity[clipped!=proposed]=0.;fixed=[]
  for i,p in enumerate(clipped):
   self.repairs+=1;q=repair(p,self.rng);q,reasons=postcheck(q,self.positions[i],self.subset,self.job['scene'])
   if reasons:self.rollback_count+=1
   for reason in reasons:self.reasons[reason]+=1
   fixed.append(q)
  fixed=np.asarray(fixed);self.velocity[fixed!=clipped]=0.;return fixed
 def accept(self,pop,values,g):
  better=values>self.scores;self.personal[better]=pop[better];self.scores[better]=values[better];idx=int(np.argmax(self.scores))
  if self.scores[idx]>self.best_score:self.best_score=float(self.scores[idx]);self.best=self.personal[idx].copy()
  self.positions=pop;self.generation=g;self.history.append(dict(generation=g,best_P0=self.best_score,mean_P0=float(values.mean()),rollback_count=self.rollback_count,repair_proposals=self.repairs))
 def save(self):
  obj={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in self.__dict__.items() if k!='rng'};obj['RNG_state']=copy.deepcopy(self.rng.bit_generator.state);obj['pbest']=self.personal.tolist();obj['gbest']=self.best.tolist();obj['velocities']=self.velocity.tolist();obj['checkpoint_utc']=now();dump(f"checkpoints/{self.job['id']}.json",obj)
 @classmethod
 def restore(cls,path):
  obj=cls.__new__(cls);data=load(path)
  for k,v in data.items():
   if k not in ['RNG_state','pbest','gbest','velocities','checkpoint_utc']:setattr(obj,k,np.array(v) if k in ['lo','hi','positions','velocity','personal','scores','best'] else v)
  obj.rng=np.random.Generator(np.random.PCG64());obj.rng.bit_generator.state=data['RNG_state'];return obj
def job(phase,name,pair,seed):
 cfg=load('configs/STEP5A_PROTOCOL.json');settings=cfg['K6' if phase=='K6' else ('stageA' if phase=='StageA' else 'stageB')];sid='ALL' if pair is None else pair['pair_id']
 return dict(id=f'{phase}_{name}_{sid}_O{seed}',phase=phase,scene=name,pair_id=sid,subset=None if pair is None else pair['nozzle_indices'],optimizer_seed=seed,physical_seed=load('configs/SEEDS_FROZEN.json')['search_physical'][name],particles=settings['particles'],updates=settings['updates'])
def run_job(j):
 final=f"layouts/{j['id']}_endpoint.json"
 if (ROOT/final).exists():check_frozen(final);return load(final)
 reserve_run(j['id'],j['phase']);cp=f"checkpoints/{j['id']}.json";s=State.restore(cp) if (ROOT/cp).exists() else State(j);assert s.job==j;begin=time.perf_counter()
 try:
  for g in range(s.generation+1,j['updates']+1):
   proposed=s.propose(g);values=[]
   for i,p in enumerate(proposed):
    if s.subset is not None:assert in_local(p,s.subset),'Frozen nozzle/R2 check before every physical layout'
    label=f"{j['id']}_G{g:02}_P{i:02}";r=evaluate(label,p,j['scene'],'P0',j['physical_seed'],j['phase']);values.append(r['coverage']);s.archive.append(dict(label=label,generation=g,particle_index=i,particle=p.tolist(),P0=r['coverage'],direct_id=r['direct_id']))
   s.accept(proposed,np.asarray(values),g);s.save();dump(f"logs/{j['id']}_progress.json",dict(id=j['id'],scene=j['scene'],phase=j['phase'],generation=g,updates=j['updates'],P0_best=s.best_score,rollback=s.rollback_count,proposals=s.repairs,utc=now()))
  ratio=s.rollback_count/s.repairs if s.repairs else 0.;result=dict(**j,particle=s.best.tolist(),P0_score=s.best_score,completed=True,rollback_count=s.rollback_count,rollback_rate=ratio,repair_proposals=s.repairs,rollback_reasons=s.reasons,local_boundary_conflict=ratio>.2,requests=len(s.archive),wall_s=time.perf_counter()-begin,completed_utc=now(),checkpoint_sha256=sha(ROOT/cp));freeze(final,result)
  with db() as c:c.execute('UPDATE runs SET status=? WHERE id=?',('COMPLETED',j['id']))
  print(f"{j['id']} complete P0={s.best_score:.7f} rollback={ratio:.3f}",flush=True);return result
 except Exception as exc:
  dump(f"logs/{j['id']}_failure.json",dict(job=j,error=repr(exc),generation=s.generation,utc=now()));raise
