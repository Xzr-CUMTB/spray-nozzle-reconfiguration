import sys,os
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from common import *
from evaluator import evaluate
from scene_model import BASELINE
def freeze_endpoints():
 check_frozen('configs/FORMAL_JOBS_FROZEN.json');check_frozen('layouts/PRIMARY_K2_ENDPOINTS_FROZEN.json');assert load('logs/formal_complete.json')['K2_runs']==96 and load('logs/formal_complete.json')['K6_runs']==32
 with db() as c:assert c.execute("SELECT COUNT(*) FROM calls WHERE fidelity='P4'").fetchone()[0]==0
 primary=load('layouts/PRIMARY_K2_ENDPOINTS_FROZEN.json')['rows'];primary_ids={r['endpoint_id'] for r in primary};representative_seed=load('configs/STEP5A_PROTOCOL.json')['representative_rule']['optimizer_seed'];rows=[]
 for name in ['V1','V2','V3','V4']:rows.append(dict(endpoint_id=f'K0_{name}',scene=name,method='K0',optimizer_seed=None,pair_id='NONE',particle=BASELINE.tolist(),primary=True,representative=True,P0_score=None))
 for job in load('configs/FORMAL_JOBS_FROZEN.json'):
  path=f"layouts/{job['id']}_endpoint.json";check_frozen(path);result=load(path);assert result['completed'];is_primary=job['phase']=='K6' or job['id'] in primary_ids
  rows.append(dict(endpoint_id=job['id'],scene=job['scene'],method=job['phase'],optimizer_seed=job['optimizer_seed'],pair_id=job['pair_id'],particle=result['particle'],primary=is_primary,representative=is_primary and job['optimizer_seed']==representative_seed,P0_score=result['P0_score'],endpoint_file_sha256=sha(ROOT/path)))
 assert len(rows)==132
 rows.append(dict(endpoint_id='EMPTY_C001',scene='EMPTY',method='EMPTY',optimizer_seed=None,pair_id='NONE',particle=BASELINE.tolist(),primary=False,representative=False,P0_score=None))
 freeze('layouts/ALL_TERMINAL_ENDPOINTS_FROZEN.json',dict(frozen_utc=now(),before_any_P4=True,endpoint_count_equipment=132,endpoint_count_with_empty=133,rows=rows));return rows
def work(row,seed):
 first=load('configs/SEEDS_FROZEN.json')['terminal_physical'][0];label=f"P4_{row['endpoint_id']}_F{seed}";r=evaluate(label,row['particle'],row['scene'],'P4',seed,'terminal',save_mask=row['representative'] and seed==first)
 return dict(endpoint_id=row['endpoint_id'],scene=row['scene'],method=row['method'],optimizer_seed=row['optimizer_seed'],pair_id=row['pair_id'],primary=row['primary'],physical_seed=seed,coverage=r['coverage'],collision_fraction=r['fractions']['collision'],active_fraction=r['fractions']['active'],free_target_volume_m3=r['free_target_volume_m3'],visited_volume_m3=r['visited_volume_m3'],status=r['status'],direct_id=r['direct_id'],label=label,wall_s=r['wall_s'],baseline_component_hits=r['baseline_component_hits'],new_nozzle_trajectory_calls=r['new_nozzle_trajectory_calls'])
def main():
 cfg=load('configs/SEEDS_FROZEN.json')
 if (ROOT/'layouts/ALL_TERMINAL_ENDPOINTS_FROZEN.json').exists():
  check_frozen('layouts/ALL_TERMINAL_ENDPOINTS_FROZEN.json');rows=load('layouts/ALL_TERMINAL_ENDPOINTS_FROZEN.json')['rows']
 else:rows=freeze_endpoints()
 out=[]
 with ProcessPoolExecutor(max_workers=load('configs/STAGE_BUDGET.json')['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
  futures=[pool.submit(work,row,seed) for row in rows for seed in cfg['terminal_physical']]
  for f in as_completed(futures):
   out.append(f.result());dump('logs/terminal_progress.json',dict(completed=len(out),expected=665,budget=snapshot()))
   if len(out)%10==0:print(f'P4 TERMINAL {len(out)}/665',flush=True)
 out=sorted(out,key=lambda r:(r['scene'],r['endpoint_id'],r['physical_seed']));assert len(out)==665
 table('P4_terminal_all.csv',[r for r in out if r['method']!='EMPTY']);table('empty_C001_reference.csv',[r for r in out if r['method']=='EMPTY']);table('K0_results.csv',[r for r in out if r['method']=='K0']);dump('logs/terminal_complete.json',dict(P4_equipment=660,P4_empty=5,all_statuses_complete=all(r['status'] in ['EVENT_COMPLETE','VOLUME_TOLERANCE_COMPLETE'] for r in out),completed_utc=now()));print('P4 TERMINAL COMPLETE 665',flush=True)
if __name__=='__main__':main()
