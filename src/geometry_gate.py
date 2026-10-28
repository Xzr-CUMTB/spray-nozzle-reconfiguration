import sys
sys.dont_write_bytecode=True
import copy,itertools
from common import *
from frozen_geometry import decode,feasible as base_feasible,LOWER,UPPER
def boxes(scene):return np.array([[b['lower'],b['upper']] for b in scene['solid_geometry']],float).reshape(-1,2,3)
def engineering_feasible(p,scene):
 if not base_feasible(p):return False
 xyz=np.array([r['position'] for r in decode(p)]);clearance=scene['mounting']['clearance']
 for box in boxes(scene):
  if np.any(np.all((xyz>=box[0]-clearance)&(xyz<=box[1]+clearance),axis=1)):return False
 for b in scene['forbidden_mounting_regions']:
  if np.any(np.all((xyz>=b['lower'])&(xyz<=b['upper']),axis=1)):return False
 return True
def union_volume(a,target=None):
 a=np.asarray(a).copy()
 if target is not None:
  a[:,0]=np.maximum(a[:,0],target[0]);a[:,1]=np.minimum(a[:,1],target[1]);a=a[np.all(a[:,1]>a[:,0],axis=1)]
 edges=[np.unique(a[:,:,d]) for d in range(3)];total=0.
 for ix in itertools.product(*[range(len(e)-1) for e in edges]):
  lo=np.array([edges[d][ix[d]] for d in range(3)]);hi=np.array([edges[d][ix[d]+1] for d in range(3)]);mid=(lo+hi)/2
  if np.any(np.all((mid>=a[:,0])&(mid<=a[:,1]),axis=1)):total+=float(np.prod(hi-lo))
 return total
def audit(scene,nominal,delta,baseline):
 a=boxes(scene);old=boxes(nominal);domain=np.array(scene['domain']);target=np.array(scene['target']);checks={}
 checks['all_AABBs_positive']=bool(np.isfinite(a).all() and np.all(a[:,1]>a[:,0]))
 checks['all_solids_inside_domain']=bool(np.all(a[:,0]>=domain[0]-1e-12) and np.all(a[:,1]<=domain[1]+1e-12))
 checks['target_valid_unchanged']=bool(np.all(target[1]>target[0]) and np.all(target[0]>=domain[0]) and np.all(target[1]<=domain[1]) and scene['target']==nominal['target'])
 checks['rigid_translation_all_components']=bool(np.allclose(a,old+np.array(delta),atol=1e-12,rtol=0) and np.allclose(a[:,1]-a[:,0],old[:,1]-old[:,0],atol=1e-12,rtol=0))
 checks['mounting_unchanged']=scene['mounting']==nominal['mounting']
 checks['forbidden_regions_translated_together']=len(scene['forbidden_mounting_regions'])==len(nominal['forbidden_mounting_regions']) and all(np.allclose(np.array([b['lower'],b['upper']]),np.array([o['lower'],o['upper']])+delta,atol=1e-12,rtol=0) for b,o in zip(scene['forbidden_mounting_regions'],nominal['forbidden_mounting_regions']))
 # Closed contact graph must connect every component to the floor through a leg.
 supported={i for i,b in enumerate(a) if abs(b[0,2]-domain[0,2])<=1e-12};changed=True
 while changed:
  old_count=len(supported)
  for i in range(len(a)):
   if any(np.all(np.maximum(a[i,0],a[j,0])<=np.minimum(a[i,1],a[j,1])+1e-12) for j in list(supported)):supported.add(i)
  changed=len(supported)>old_count
 checks['all_components_ground_connected']=len(supported)==len(a)
 v=union_volume(a);tv=union_volume(a,target);checks['positive_solid_union']=v>0
 checks['C001_installation_feasible']=engineering_feasible(baseline,scene)
 xyz=np.array([r['position'] for r in decode(baseline)]);spacing=min(np.linalg.norm(xyz[i]-xyz[j]) for i,j in itertools.combinations(range(6),2));checks['C001_spacing_ge_030']=spacing>=.3
 checks={k:bool(value) for k,value in checks.items()}
 return dict(scene=scene['id'],passed=all(checks.values()),checks=checks,solid_union_m3=v,target_solid_union_m3=tv,free_target_m3=float(np.prod(target[1]-target[0])-tv),min_C001_spacing_m=float(spacing),solid_count=len(a),explicit_forbidden_box_count=len(scene['forbidden_mounting_regions']),implicit_clearance_m=scene['mounting']['clearance'],physics_calls=0)
