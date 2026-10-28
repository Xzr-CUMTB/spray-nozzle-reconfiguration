import sys,os
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import copy
from common import *
from scene_model import *
from search import State,job,postcheck
from frozen_pso import optimize
def exact(a,b):assert np.asarray(a,dtype=np.float64).tobytes()==np.asarray(b,dtype=np.float64).tobytes()
def main():
 assert not (ROOT/'logs/direct_calls.sqlite').exists()
 for name in ['V1','V2','V3','V4']:
  assert engineering_feasible(BASELINE,scene(name))
  for h in [.1,.04]:assert abs(float(free_volume(name,h).sum())-71.47536)<1e-10
 # Full bounds PSO must match the frozen STEP5C update on an analytic fixture.
 j=job('K6','V1',None,2026101201);j.update(id='TEST_ZERO_PHYSICS_FULL',particles=4,updates=3)
 state=State(j);positions=state.positions.copy();rng=np.random.Generator(np.random.PCG64());rng.bit_generator.state=copy.deepcopy(state.rng.bit_generator.state);observed=[]
 def synthetic(pop,g):return -np.sum((pop-BASELINE)**2,axis=1)
 optimize(positions,rng,synthetic,3,lambda g,pop,values,*args:observed.append((pop.copy(),values.copy())))
 for g in range(4):
  pop=state.propose(g);values=synthetic(pop,g);exact(pop,observed[g][0]);exact(values,observed[g][1]);state.accept(pop,values,g)
  if g==1:state.save();state=State.restore(f"checkpoints/{j['id']}.json")
 assert state.rng.bit_generator.state==rng.bit_generator.state
 pair=dict(pair_id='P12',nozzle_indices=[0,1]);j=job('StageA','V1',pair,2026101101);j.update(id='TEST_ZERO_PHYSICS_LOCAL',updates=4);state=State(j)
 for g in range(5):
  p=state.propose(g);assert all(in_local(row,[0,1]) for row in p);state.accept(p,synthetic(p,g),g)
 previous=BASELINE.copy();q=previous.copy();q[13]+=1e-8;fixed,reasons=postcheck(q,previous,[0,1],'V1');exact(fixed,previous);assert 'frozen_parameter_changed' in reasons
 q=previous.copy();q[1]=local_bounds([0,1])[1][1]+.01;fixed,reasons=postcheck(q,previous,[0,1],'V1');exact(fixed,previous);assert 'selected_radius_violation' in reasons
 # Definitions of true-axis change: identical axes must produce exactly zero.
 nozzles=decode(BASELINE);axes=[]
 for n in nozzles:
  yaw,pitch=np.deg2rad([n['yaw_deg'],n['pitch_deg']]);axes.append([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
 axes=np.array(axes);assert np.max(np.degrees(np.arctan2(np.linalg.norm(np.cross(axes,axes),axis=1),np.sum(axes*axes,axis=1))))==0
 dump('logs/preflight_tests.json',dict(passed=True,physics_calls=0,scene_geometry_and_exact_free_volume=True,full_PSO_positions_scores_RNG_match=True,checkpoint_resume_exact=True,frozen_nozzle_checks=True,postcheck_rejection_tests=True,true_axis_identity=True,utc=now()))
 print('ZERO-PHYSICS PREFLIGHT PASSED')
if __name__=='__main__':main()
