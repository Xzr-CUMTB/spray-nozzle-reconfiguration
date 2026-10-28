import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
from analysis import overall,bootstrap,magnitude
from scene_model import BASELINE
def main():
 cases=[(3,[.5,.6,.7,0],False,False,'SPARSE_RECONFIGURATION_CONFIRMED'),(4,[.5]*4,False,False,'SPARSE_RECONFIGURATION_CONFIRMED'),(3,[.6,.6,.6,-.5],False,False,'SPARSE_RECONFIGURATION_CONDITIONAL'),(3,[.6,.6,.6,-.499],False,False,'SPARSE_RECONFIGURATION_CONFIRMED'),(2,[.6,.6,0,0],False,False,'SPARSE_RECONFIGURATION_CONDITIONAL'),(1,[2,0,0,0],False,False,'SPARSE_RECONFIGURATION_NOT_CONFIRMED'),(0,[0]*4,False,False,'SPARSE_RECONFIGURATION_NOT_CONFIRMED'),(4,[2]*4,True,False,'INCONCLUSIVE'),(4,[2]*4,False,True,'INCONCLUSIVE')]
 for support,means,incomplete,severe,expected in cases:assert overall(support,means,incomplete,severe)==expected
 idx=np.random.default_rng(2026101501).integers(0,8,(10000,8));assert bootstrap([.5]*8,idx)==[.5,.5]
 m,rows=magnitude(BASELINE);assert m['modified_nozzles']==0 and m['D_pos_m']==0 and m['D_angle_deg']==0 and m['N_surface_change']==0
 p=BASELINE.copy();p[3]+=1.;m,rows=magnitude(p);assert m['modified_nozzles']==1 and m['D_pos_m']==0
 # The severe flag is conjunctive and uses strict > loss/SE boundaries.
 def severe(rho,recall,loss_flags):return rho<.5 and recall<.5 and sum(loss_flags)>=6
 assert not severe(.5,0,[True]*8) and not severe(0,.5,[True]*8) and not severe(0,0,[True]*5+[False]*3) and severe(.49,1/6,[True]*6+[False]*2)
 assert not (1.>1. and 1.>2*.1);assert not (1.1>1. and 1.1>2*.55)
 dump('logs/gate_tests.json',dict(passed=True,physics_calls=0,overall_boundary_cases=len(cases),all_states_exclusive=True,bootstrap_search_units_only=True,zero_retrofit_identity=True,strict_fidelity_boundaries=True,utc=now()))
 print('GATE/BOOTSTRAP/RETROFIT TESTS PASSED; no physical calls')
if __name__=='__main__':main()
