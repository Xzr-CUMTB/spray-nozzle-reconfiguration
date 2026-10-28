import sys
sys.dont_write_bytecode=True
from common import *
from scipy.stats import spearmanr
from scene_model import BASELINE,decode
def readcsv(path):
 with (ROOT/path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def summary(values):
 a=np.array(values,float);return dict(mean=float(a.mean()),SD=float(a.std(ddof=1)),median=float(np.median(a)),min=float(a.min()),max=float(a.max()))
def bootstrap(values,indices):return [float(x) for x in np.quantile(np.array(values)[indices].mean(axis=1),[.025,.975],method='linear')]
def overall(support,means,incomplete=False,severe=False):
 if incomplete or severe:return 'INCONCLUSIVE'
 harm=any(v<=-.5 for v in means)
 if support>=3 and not harm:return 'SPARSE_RECONFIGURATION_CONFIRMED'
 if support==2 or (support>=3 and harm):return 'SPARSE_RECONFIGURATION_CONDITIONAL'
 return 'SPARSE_RECONFIGURATION_NOT_CONFIRMED'
def axes(p):
 out=[]
 for n in decode(p):
  yaw,pitch=np.deg2rad([n['yaw_deg'],n['pitch_deg']]);out.append([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
 return np.array(out)
def magnitude(p):
 p=np.array(p);a=p.reshape(6,5);b=BASELINE.reshape(6,5);modified=np.any(a!=b,axis=1);xyz=np.array([r['position'] for r in decode(p)]);base_xyz=np.array([r['position'] for r in decode(BASELINE)]);position=np.linalg.norm(xyz-base_xyz,axis=1);x,y=axes(p),axes(BASELINE);angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(x,y),axis=1),np.sum(x*y,axis=1)));surface=np.floor(a[:,0])!=np.floor(b[:,0])
 return dict(modified_nozzles=int(modified.sum()),modified_ids='-'.join(str(i+1) for i in np.flatnonzero(modified)),D_pos_m=float(position[modified].sum()),D_angle_deg=float(angle[modified].sum()),N_surface_change=int(surface.sum())),[dict(nozzle=i+1,modified=bool(modified[i]),D_pos_m=float(position[i]),D_angle_deg=float(angle[i]),surface_changed=bool(surface[i]),before_json=json.dumps(b[i].tolist()),after_json=json.dumps(a[i].tolist())) for i in range(6)]
def main():
 cfg=load('configs/STEP5A_PROTOCOL.json');seeds=load('configs/SEEDS_FROZEN.json');raw=readcsv('P4_terminal_all.csv');empty=readcsv('empty_C001_reference.csv');prim=load('layouts/PRIMARY_K2_ENDPOINTS_FROZEN.json')['rows'];endpoints=load('layouts/ALL_TERMINAL_ENDPOINTS_FROZEN.json')['rows'];lookup={r['endpoint_id']:r for r in endpoints};nphys=5
 def values(ident):
  rr=sorted([r for r in raw+empty if r['endpoint_id']==ident],key=lambda r:int(r['physical_seed']));assert [int(r['physical_seed']) for r in rr]==seeds['terminal_physical'];return np.array([float(r['coverage']) for r in rr])
 rng=np.random.default_rng(seeds['bootstrap']);indices=rng.integers(0,8,(10000,8));np.save(ROOT/'configs/bootstrap_indices.npy',indices)
 comparisons=[];effects=[];gaps=[];magnitudes=[];nozzles=[];obstruction=[];scene_rows=[];fidelity=[];frequency=[];all_losses=[];raw_means=[]
 for endpoint in endpoints:
  if endpoint['method']=='EMPTY':continue
  vv=values(endpoint['endpoint_id']);raw_means.append(dict(endpoint_id=endpoint['endpoint_id'],scene=endpoint['scene'],method=endpoint['method'],optimizer_seed=endpoint['optimizer_seed'],pair_id=endpoint['pair_id'],primary=endpoint['primary'],P0_score=endpoint['P0_score'],P4_mean=float(vv.mean()),P4_SD_physical=float(vv.std(ddof=1))))
 cempty=values('EMPTY_C001')
 for name in cfg['scenes']:
  c0=values('K0_'+name);loss=100*(c0-cempty)
  for i,phy in enumerate(seeds['terminal_physical']):obstruction.append(dict(scene=name,physical_seed=phy,C_empty=cempty[i],C_K0=c0[i],L_obstruction_pp=loss[i]))
  gains=[];g26=[];physical_rule=[];c2means=[];c6means=[];pair_counts={p['pair_id']:0 for p in cfg['pairs']};nozzle_counts=np.zeros(6,int);loss_qualifiers=0
  for search_seed in seeds['formal_optimizer']:
   primary=next(r for r in prim if r['scene']==name and r['optimizer_seed']==search_seed);id2=primary['endpoint_id'];id6=f'K6_{name}_ALL_O{search_seed}';c2=values(id2);c6=values(id6);delta=100*(c2-c0);gap=100*(c6-c2);mean_gain=float(delta.mean());gains.append(mean_gain);g26.append(float(gap.mean()));npositive=int(np.count_nonzero(delta>0));physical_rule.append(npositive>=4);c2means.append(float(c2.mean()));c6means.append(float(c6.mean()))
   for i,phy in enumerate(seeds['terminal_physical']):comparisons.append(dict(scene=name,optimizer_seed=search_seed,physical_seed=phy,K2_endpoint_id=id2,K6_endpoint_id=id6,pair_id=primary['pair_id'],C0=float(c0[i]),C2=float(c2[i]),C6=float(c6[i]),delta2_pp=float(delta[i]),gap26_pp=float(gap[i])))
   effects.append(dict(scene=name,optimizer_seed=search_seed,K2_endpoint_id=id2,pair_id=primary['pair_id'],C0_mean=float(c0.mean()),C2_mean=float(c2.mean()),mean_delta2_pp=mean_gain,physical_SD_delta2_pp=float(delta.std(ddof=1)),physical_SE_delta2_pp=float(delta.std(ddof=1)/np.sqrt(5)),positive_physical_seeds=npositive))
   gaps.append(dict(scene=name,optimizer_seed=search_seed,C6_mean=float(c6.mean()),C2_mean=float(c2.mean()),G26_mean_pp=float(gap.mean()),physical_SD_G26_pp=float(gap.std(ddof=1)),G26_le_050pp=bool(gap.mean()<=.5)))
   mag,details=magnitude(lookup[id2]['particle']);assert mag['modified_nozzles']<=2;magnitudes.append(dict(scene=name,optimizer_seed=search_seed,endpoint_id=id2,pair_id=primary['pair_id'],mean_delta2_pp=mean_gain,**mag));nozzles.extend(dict(scene=name,optimizer_seed=search_seed,endpoint_id=id2,**d) for d in details);pair_counts[primary['pair_id']]+=1
   for digit in primary['pair_id'][1:]:nozzle_counts[int(digit)-1]+=1
   candidates=[e for e in endpoints if e['scene']==name and e['method']=='K2' and e['optimizer_seed']==search_seed];p4_best=min(candidates,key=lambda e:(-float(values(e['endpoint_id']).mean()),e['endpoint_id']));bestval=values(p4_best['endpoint_id']);diff=100*(bestval-c2);ls=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(5));qualifies=ls>1. and ls>2*se;loss_qualifiers+=int(qualifies);all_losses.append(dict(scene=name,optimizer_seed=search_seed,P0_primary=id2,P4_best_secondary_only=p4_best['endpoint_id'],loss_pp=ls,paired_SE_pp=se,severe_loss_condition=qualifies))
  s=summary(gains);ci=bootstrap(gains,indices);supported=s['mean']>=.5 and sum(v>0 for v in gains)>=6 and sum(physical_rule)>=6 and ci[0]>0
  scene_rows.append(dict(scene=name,status='K2_GAIN_SUPPORTED' if supported else 'K2_GAIN_NOT_SUPPORTED',supported=supported,mean_delta2_pp=s['mean'],SD_delta2_across_optimizer_pp=s['SD'],median_delta2_pp=s['median'],min_delta2_pp=s['min'],max_delta2_pp=s['max'],bootstrap95_low_pp=ci[0],bootstrap95_high_pp=ci[1],positive_optimizer_seeds=sum(v>0 for v in gains),optimizer_seeds_with_ge4_positive_physical=sum(physical_rule),K0_P4_mean=float(c0.mean()),K0_P4_SD_physical=float(c0.std(ddof=1)),K2_P4_mean=float(np.mean(c2means)),K6_P4_mean=float(np.mean(c6means)),mean_G26_pp=float(np.mean(g26)),G26_le050_optimizer_seeds=sum(v<=.5 for v in g26),mean_obstruction_loss_pp=float(loss.mean()),physical_obstruction_SD_pp=float(loss.std(ddof=1)),negative_delta2_optimizer_seeds=sum(v<0 for v in gains)))
  rr=[r for r in raw_means if r['scene']==name and r['method']=='K2'];assert len(rr)==24
  po=sorted(rr,key=lambda r:(-r['P0_score'],r['endpoint_id']));hi=sorted(rr,key=lambda r:(-r['P4_mean'],r['endpoint_id']));ids=sorted(r['endpoint_id'] for r in rr);pr={r['endpoint_id']:i+1 for i,r in enumerate(po)};hr={r['endpoint_id']:i+1 for i,r in enumerate(hi)};evaluable=len({r['P0_score'] for r in rr})>1 and len({r['P4_mean'] for r in rr})>1
  rho=float(spearmanr([pr[i] for i in ids],[hr[i] for i in ids]).statistic) if evaluable else None;recall=len({r['endpoint_id'] for r in po[:6]}&{r['endpoint_id'] for r in hi[:6]})/6;severe=evaluable and rho<.5 and recall<.5 and loss_qualifiers>=6
  fidelity.append(dict(scene=name,n_candidates=24,Spearman=rho,top25_recall=recall,optimizer_seeds_severe_pair_loss=loss_qualifiers,severe=bool(severe),status='SEVERE_HIGH_VALUE_FIDELITY_MISMATCH' if severe else ('NO_SEVERE_MISMATCH' if evaluable else 'DIAGNOSTIC_NOT_EVALUABLE'),primary_reselected_by_P4=False))
  for p,count in pair_counts.items():frequency.append(dict(scene=name,type='pair',identity=p,count=count))
  for i,count in enumerate(nozzle_counts):frequency.append(dict(scene=name,type='nozzle',identity=f'N{i+1}',count=int(count)))
  for i in range(6):frequency.append(dict(scene=name,type='actually_modified_nozzle',identity=f'N{i+1}',count=sum(r['scene']==name and r['nozzle']==i+1 and r['modified'] for r in nozzles)))
 table('P4_endpoint_summary.csv',raw_means);table('P4_primary_comparison.csv',comparisons);table('primary_effects_by_search_seed.csv',effects);table('primary_effects.csv',scene_rows);table('remaining_gap.csv',gaps);table('retrofit_magnitude.csv',magnitudes);table('retrofit_per_nozzle.csv',nozzles);table('obstruction_loss.csv',obstruction);table('subset_selection_frequency.csv',frequency);table('local_fidelity_diagnostics.csv',fidelity);table('local_fidelity_pair_losses.csv',all_losses)
 fluctuations=[]
 for name in cfg['scenes']:
  means=[np.mean([r['delta2_pp'] for r in comparisons if r['scene']==name and r['physical_seed']==seed]) for seed in seeds['terminal_physical']];fluctuations.append(dict(scene=name,mean_gain_pp=float(np.mean(means)),physical_SD_of_search_mean_gain_pp=float(np.std(means,ddof=1)),physical_SE_of_search_mean_gain_pp=float(np.std(means,ddof=1)/np.sqrt(5)),five_physical_seed_search_averaged_gains_json=json.dumps(means),interpretation='secondary physical sampling fluctuation, not additional optimizer replicates'))
 table('physical_seed_fluctuation.csv',fluctuations)
 directional=[]
 for left,right in [('V1','V2'),('V3','V4')]:
  a=np.array([r['mean_delta2_pp'] for r in effects if r['scene']==left]);b=np.array([r['mean_delta2_pp'] for r in effects if r['scene']==right]);ci=bootstrap(a-b,indices);directional.append(dict(contrast=f'{left}-{right}',mean_difference_pp=float(np.mean(a-b)),bootstrap95_low_pp=ci[0],bootstrap95_high_pp=ci[1],changes_Gate=False))
 table('directional_secondary.csv',directional);support=sum(r['supported'] for r in scene_rows);status=overall(support,[r['mean_delta2_pp'] for r in scene_rows],severe=any(r['severe'] for r in fidelity))
 dump('STATISTICAL_RESULTS.json',dict(status=status,supported_scenes=support,scenes=scene_rows,fidelity=fidelity,unit='8 optimizer search seeds, each endpoint averaged over five paired physical seeds',bootstrap_replicates=10000,bootstrap_indices_sha256=sha(ROOT/'configs/bootstrap_indices.npy'),primary_K2_unchanged=True,C001_unchanged=True,computed_utc=now()));print('STATISTICAL_GATE',status,json.dumps(scene_rows),flush=True)
if __name__=='__main__':main()
