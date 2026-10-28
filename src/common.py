import sys,os
sys.dont_write_bytecode=True
import json,csv,hashlib,datetime,sqlite3,time
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('SPRAY_RUN_DIR', Path(__file__).resolve().parents[1]/'runs'/'default')).resolve();BASE=ROOT.parent
def load(p):return json.loads((ROOT/p).read_text('utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def elapsed():return (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(load('configs/STAGE_BUDGET.json')['start_utc'])).total_seconds()
def dump(p,x):
 dest=ROOT/p;dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(dest)
def table(p,rows,fields=None):
 dest=ROOT/p;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def freeze(p,x):
 dest=ROOT/p;assert not dest.exists(),f'Already frozen: {p}';dump(p,x);dest.with_suffix(dest.suffix+'.sha256').write_text(sha(dest),encoding='ascii');return sha(dest)
def check_frozen(p):assert sha(ROOT/p)==(ROOT/p).with_suffix((ROOT/p).suffix+'.sha256').read_text()
def db():
 c=sqlite3.connect(ROOT/'logs/direct_calls.sqlite',timeout=60);c.execute('PRAGMA busy_timeout=60000');return c
def initialize_ledger():
 with db() as c:
  c.execute('PRAGMA journal_mode=WAL');c.execute('CREATE TABLE IF NOT EXISTS counters(category TEXT PRIMARY KEY,n INTEGER NOT NULL)');c.execute('CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY AUTOINCREMENT,label TEXT UNIQUE,phase TEXT,scene TEXT,fidelity TEXT,seed INTEGER,particle_sha256 TEXT,status TEXT,started_utc TEXT,ended_utc TEXT,wall_s REAL,error TEXT)');c.execute('CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,phase TEXT,status TEXT)')
  for k in ['P0','P4','all_direct','formal_K2_runs','formal_K6_runs']:c.execute('INSERT OR IGNORE INTO counters VALUES(?,0)',(k,))
def reserve_run(ident,phase):
 with db() as c:
  c.execute('BEGIN IMMEDIATE');existing=c.execute('SELECT phase,status FROM runs WHERE id=?',(ident,)).fetchone()
  if existing:return
  if phase in ['K2','K6']:
   category=f'formal_{phase}_runs';limit=load('configs/STAGE_BUDGET.json')[category];n=c.execute('SELECT n FROM counters WHERE category=?',(category,)).fetchone()[0];assert n<limit; c.execute('UPDATE counters SET n=n+1 WHERE category=?',(category,))
  c.execute('INSERT INTO runs VALUES(?,?,?)',(ident,phase,'STARTED'))
def reserve(label,phase,scene,fidelity,seed,digest):
 budget=load('configs/STAGE_BUDGET.json');assert elapsed()<budget['stop_new_physics_s'],'WALL_BUDGET_STOP'
 with db() as c:
  c.execute('BEGIN IMMEDIATE')
  assert c.execute('SELECT id FROM calls WHERE label=?',(label,)).fetchone() is None,'No repeat of an unfinished direct request'
  for category in ['all_direct',fidelity]:
   n=c.execute('SELECT n FROM counters WHERE category=?',(category,)).fetchone()[0];assert n<budget[category],f'{category}_BUDGET_STOP';c.execute('UPDATE counters SET n=n+1 WHERE category=?',(category,))
  cur=c.execute('INSERT INTO calls(label,phase,scene,fidelity,seed,particle_sha256,status,started_utc) VALUES(?,?,?,?,?,?,?,?)',(label,phase,scene,fidelity,int(seed),digest,'STARTED',now()));return cur.lastrowid
def finish(ident,seconds,error=None):
 with db() as c:c.execute('UPDATE calls SET status=?,ended_utc=?,wall_s=?,error=? WHERE id=?',('FAILED' if error else 'COMPLETED',now(),seconds,error,ident))
def snapshot():
 with db() as c:
  counts=dict(c.execute('SELECT category,n FROM counters'));status=[dict(phase=a,status=b,n=n) for a,b,n in c.execute('SELECT phase,status,COUNT(*) FROM calls GROUP BY phase,status')]
 return dict(counts=counts,status=status,wall_s=elapsed(),utc=now())
