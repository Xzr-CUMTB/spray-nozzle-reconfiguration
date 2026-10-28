import sys,os
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from common import *
from search import job,run_job
def batch(jobs,phase):
 results=[]
 with ProcessPoolExecutor(max_workers=load('configs/STAGE_BUDGET.json')['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
  futures=[pool.submit(run_job,j) for j in jobs]
  for f in as_completed(futures):
   results.append(f.result());dump(f'logs/{phase}_progress.json',dict(completed=len(results),expected=len(jobs),last=results[-1]['id'],budget=snapshot()))
 return sorted(results,key=lambda r:r['id'])
def stageA():
 cfg=load('configs/STEP5A_PROTOCOL.json');seeds=load('configs/SEEDS_FROZEN.json');jobs=[job('StageA',name,pair,seed) for name in cfg['scenes'] for pair in cfg['pairs'] for seed in seeds['screening_optimizer']]
 freeze('configs/StageA_jobs.json',jobs);results=batch(jobs,'StageA');ranking=[];selected={}
 for name in cfg['scenes']:
  rows=[]
  for p in cfg['pairs']:
   rr=[r for r in results if r['scene']==name and r['pair_id']==p['pair_id']];assert len(rr)==2;values=[r['P0_score'] for r in rr];rows.append(dict(scene=name,pair_id=p['pair_id'],subset=p['nozzle_indices'],screening_scores=values,median_best=float(np.median(values)),best=float(max(values)),run_ids=[r['id'] for r in rr]))
  rows=sorted(rows,key=lambda r:(-r['median_best'],-r['best'],r['pair_id']));selected[name]=[]
  for i,r in enumerate(rows,1):r.update(rank=i,selected=i<=3);ranking.append(r)
  selected[name]=[dict(pair_id=r['pair_id'],nozzle_indices=r['subset']) for r in rows[:3]]
 freeze('layouts/TOP3_PAIRS_FROZEN.json',dict(selected=selected,full_rankings=ranking,frozen_utc=now(),before_StageB=True));table('subset_screening.csv',[{**r,'subset':json.dumps(r['subset']),'screening_scores':json.dumps(r['screening_scores']),'run_ids':json.dumps(r['run_ids'])} for r in ranking]);dump('logs/StageA_complete.json',dict(runs=len(results),requests=sum(r['requests'] for r in results),utc=now()));print('STAGE_A_COMPLETE',json.dumps(selected),flush=True)
def formal():
 cfg=load('configs/STEP5A_PROTOCOL.json');seeds=load('configs/SEEDS_FROZEN.json');selected=load('layouts/TOP3_PAIRS_FROZEN.json')['selected'];jobs=[]
 for name in cfg['scenes']:
  for seed in seeds['formal_optimizer']:
   jobs.extend(job('K2',name,p,seed) for p in selected[name]);jobs.append(job('K6',name,None,seed))
 assert sum(j['phase']=='K2' for j in jobs)==96 and sum(j['phase']=='K6' for j in jobs)==32
 freeze('configs/FORMAL_JOBS_FROZEN.json',jobs);results=batch(jobs,'formal');k2=[r for r in results if r['phase']=='K2'];k6=[r for r in results if r['phase']=='K6'];primary=[]
 for name in cfg['scenes']:
  for seed in seeds['formal_optimizer']:
   rr=[r for r in k2 if r['scene']==name and r['optimizer_seed']==seed];assert len(rr)==3;winner=min(rr,key=lambda r:(-r['P0_score'],r['pair_id']));primary.append(dict(scene=name,optimizer_seed=seed,endpoint_id=winner['id'],pair_id=winner['pair_id'],P0_score=winner['P0_score'],particle_sha256=hashlib.sha256(np.array(winner['particle']).tobytes()).hexdigest()))
 freeze('layouts/PRIMARY_K2_ENDPOINTS_FROZEN.json',dict(rows=primary,frozen_utc=now(),P4_results_seen=0));table('K2_primary_endpoints.csv',primary)
 fields=['id','scene','pair_id','optimizer_seed','physical_seed','P0_score','requests','rollback_count','rollback_rate','repair_proposals','local_boundary_conflict','wall_s']
 table('K2_formal_runs.csv',[{k:r[k] for k in fields} for r in k2]);table('K6_formal_runs.csv',[{k:r[k] for k in fields} for r in k6]);dump('logs/formal_complete.json',dict(K2_runs=len(k2),K6_runs=len(k6),requests=sum(r['requests'] for r in results),utc=now()));print('FORMAL_SEARCH_COMPLETE',len(k2),len(k6),flush=True)
