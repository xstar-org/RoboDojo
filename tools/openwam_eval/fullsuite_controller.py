"""One policy GPU, multiple workers on one simulator GPU; preserve native outcomes."""
import concurrent.futures,hashlib,json,os,pathlib,queue,signal,socket,subprocess,threading,time,traceback
from native_accounting import collect_attempt
C=pathlib.Path(__file__).resolve().parent
assert os.environ['SLURM_JOB_ID']==json.loads((C/'manifest.json').read_text())['job']
PLANS=json.loads((C.parent/'task-plans.json').read_text())
M=json.loads((C/'manifest.json').read_text())
CPUS=M['cpus'];assert set(CPUS)<=set(os.sched_getaffinity(0)) and len(CPUS)>=24
SIM_GPUS=M['sim_gpus'];assert 1<=len(SIM_GPUS)<=3 and M['policy_gpu'] not in SIM_GPUS
assert len(set(SIM_GPUS))==len(SIM_GPUS)
assert M['checkpoint_step']>0 and M['filesystem_preflight_passed']
assert hashlib.sha256((C.parent/'task-plans.json').read_bytes()).hexdigest()==M['task_plans_sha256']
PORT=M['port']
RUN_TAG=str(time.time_ns())
LOCK=threading.RLock();STOP=threading.Event();CHILDREN=[];STATES={};WORK=queue.Queue()
priority=['play_tic_tac_toe']
def save(p,x):
 with LOCK:
  tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def event(kind,**kw):
 with LOCK:
  with (C/'events.jsonl').open('a') as f:f.write(json.dumps(dict(t=time.time(),kind=kind,**kw))+'\n')
def progress():save(C/'progress.json',{'t':time.time(),'job':os.environ['SLURM_JOB_ID'],'tasks':STATES,'completed':sum(s.get('state')=='complete' for s in STATES.values()),'total':len(PLANS),'queued':WORK.qsize()})
def spawn(args,env,cwd,log,cpus,role):
 with log.open('w') as f:p=subprocess.Popen(['taskset','-c',','.join(map(str,cpus))]+list(map(str,args)),env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q=pathlib.Path('/proc')/str(p.pid)
 cookie={'pid':p.pid,'start':q.joinpath('stat').read_text().split(') ',1)[1].split()[19],'cgroup':q.joinpath('cgroup').read_text(),'cmdline':q.joinpath('cmdline').read_bytes().decode().replace('\0',' '),'role':role}
 with LOCK:CHILDREN.append((p,cookie));save(C/'children.json',[c for _,c in CHILDREN])
 event('spawn',**cookie);return p
def terminate(p):
 if p.poll() is not None:return
 cookie=next(c for q,c in CHILDREN if q is p);d=pathlib.Path('/proc')/str(p.pid)
 try:
  fields=d.joinpath('stat').read_text().split(') ',1)[1].split()
  if fields[0] in ['Z','X']:return
  assert fields[19]==cookie['start'] and d.joinpath('cgroup').read_text()==cookie['cgroup']
  assert '/job_'+str(M['job'])+'/' in cookie['cgroup'] and os.getpgid(p.pid)==p.pid
  os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:p.wait(20)
 except subprocess.TimeoutExpired:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  p.wait(20)
def stop_requested(signum,frame):
 STOP.set()
 event('stop_requested',signal=signum)
signal.signal(signal.SIGTERM,stop_requested)
signal.signal(signal.SIGINT,stop_requested)
def wait_all(active,policy,timeout=14400):
 start=time.time()
 while any(p.poll() is None for p,_,_ in active):
  if STOP.is_set() or policy.poll() is not None:raise RuntimeError('controller stopped or policy exited')
  for p,A,began in active:
   if p.poll() is not None:continue
   metric=A/'metrics.jsonl'
   text=metric.read_text() if metric.exists() else ''
   log=A/'client.log'
   tail=log.read_text(errors='replace')[-18000:] if log.exists() else ''
   if any(signature in tail for signature in ('[Fatal] [carb.crashreporter-breakpad.plugin]', 'Simulation cannot continue! Error code 700', '[Error] [carb.graphics-vulkan.plugin] GPU crash is detected', 'ERROR_DEVICE_LOST')):
    event('infrastructure_native_crash',pid=p.pid,log=str(log));terminate(p);continue
   latest=metric.stat().st_mtime if metric.exists() else began
   limit=1800 if '"run_start"' in text else 3600
   if time.time()-latest>limit:
    event('infrastructure_timeout',pid=p.pid,log=str(A/'client.log'));terminate(p)
  if time.time()-start>timeout:
   for p,_,_ in active:terminate(p)
   break
  time.sleep(5)
def find_result(R,task,runid):
 root=R/f'eval_result/RoboDojo/{task}/OpenWAM/arx_x5'
 files=list(root.rglob('_result.json')) if root.exists() else []
 matches=[p for p in files if p.parent.name==runid]
 assert len(matches)<=1
 return (json.loads(matches[0].read_text()),str(matches[0])) if matches else (None,None)

def recover_cases(task):
 rows={};excluded={};finished=None
 expected={(s['seed'],i) for s in PLANS[task]['seeds'] for i in s['layout_ids']}
 for G in sorted(C.glob('group*')):
  D=G/'tasks'/task
  if not D.exists():continue
  for name,target in [('partial.json',rows),('native-exclusions.json',excluded)]:
   p=D/name
   if p.exists():
    for item in json.loads(p.read_text()):
     key=(item['seed'],item['layout_id']);assert key in expected
     if key in target:assert target[key]['layout_id']==item['layout_id']
     target.setdefault(key,item)
  p=D/'result.json'
  if p.exists():
   result=json.loads(p.read_text());assert result['state']=='complete'
   assert len(result['details'])+len(result['native_exclusions'])==len(expected)
   assert finished is None,'duplicate completed task'
   finished=result
   for item in result['details']:rows.setdefault((item['seed'],item['layout_id']),item)
   for item in result['native_exclusions']:excluded.setdefault((item['seed'],item['layout_id']),item)
  for request in sorted(D.glob('seed*/a*/request.json')):
   A=request.parent
   if (A/'STRATEGY_INTERRUPTED.json').exists():continue
   q=json.loads(request.read_text());data,path=find_result(G/'RoboDojo',task,q['run_id'])
   if data is None:continue
   before=set(rows);worker=int(A.parent.name.split('-w')[1]);attempt=int(A.name[1:])
   collect_attempt(rows,excluded,worker=worker,seed=q['seed'],requested=q['layout_ids'],data=data,log=(A/'client.log').read_text(errors='replace'),attempt=attempt)
   indices={int(v['layout_id']):int(k) for k,v in data.get('details',{}).items()}
   for key in set(rows)-before:rows[key].update(result_source=path,result_index=indices[key[1]],attempt=attempt,client_log=str(A/'client.log'))
 assert not set(rows)&set(excluded)
 assert (set(rows)|set(excluded))<=expected
 if finished is not None:assert set(rows)|set(excluded)==expected
 return rows,excluded,finished


# Appended to the existing controller's native-accounting and process helpers.
from multigpu_plan import MetricSteps, next_assignment, pending_keys, slot_specs
from process_identity import ExistingProcess
CONFIG=json.loads((C.parent/'serial-multigpu-config.json').read_text())
assert M['policy_gpu'] not in SIM_GPUS and SIM_GPUS
assert M['workers_per_gpu']==3
GATE=json.loads(pathlib.Path(M['resume_gate']).read_text());assert GATE['full_resume_allowed'] is True
assert hashlib.sha256(pathlib.Path(M['resume_gate']).read_bytes()).hexdigest()==M['resume_gate_sha256']
PERF=json.loads(pathlib.Path(GATE['performance_audit']).read_text())
assert PERF['mechanical_and_quality_passed'] is True
assert GATE['resume_below100'] is True and GATE['authorization_kind']=='explicit_user_full_evaluation_priority'
from balanced_model_plan import next_assignment, balanced_slots
SLOTS=balanced_slots(M['sim_cpu_pools'],SIM_GPUS,M['workers_per_gpu'])
assert len(SLOTS)==len(SIM_GPUS)*M['workers_per_gpu'] and all(set(x['cpus'])<=set(CPUS) and not set(x['cpus'])&set(M['policy_cpus']) for x in SLOTS)
G=C/'group0';R=G/'RoboDojo';OW=R/'XPolicyLab/policy/OpenWAM'
RUN_TAG=CONFIG['run_tag']+'-'+C.name
START=time.time();ACTIVE={};TASKS={};TOTAL_STEPS=0;GPU_STEPS={str(g):0 for g in M['sim_gpus']}
METRICS={};LAST_SAMPLE=0;LAST_LAUNCH=0;FAULTED_GPUS=set()
priority=['play_tic_tac_toe']+sorted(t for t in PLANS if t!='play_tic_tac_toe')
BATCHES={t:25 for t in PLANS}
HOLDS=dict(CONFIG.get('held_tasks',{}))
priority=['play_tic_tac_toe','build_tower','align_blocks','arrange_largest_number']+[t for t in priority if t not in {'play_tic_tac_toe','build_tower','align_blocks','arrange_largest_number'}]
assert set(priority)==set(PLANS)

def progress():
 summary={t:{k:v for k,v in s.items() if k not in ['details','native_exclusions','per_seed']} for t,s in STATES.items()}
 save(C/'progress.json',{'t':time.time(),'job':M['job'],'strategy':'one_model_policy1_sim023_three_workers_each_balanced25_pt16','tasks':summary,'completed':sum(s.get('state')=='complete' for s in STATES.values()),'total':54,'queued':sum(s.get('state')=='queued' for s in STATES.values()),'active_workers':len(ACTIVE)})

def publish(task):
 d=TASKS[task];D=G/'tasks'/task;D.mkdir(parents=True,exist_ok=True)
 save(D/'partial.json',list(d['rows'].values()));save(D/'native-exclusions.json',list(d['excluded'].values()))
 if STATES[task].get('state')!='complete':
  STATES[task].update(state='running' if any(x['task']==task for x in ACTIVE.values()) else 'queued',valid=len(d['rows']),unstable=len(d['excluded']),successes=sum(v['success'] for v in d['rows'].values()),active_workers=sum(x['task']==task for x in ACTIVE.values()))

def harvest(item):
 task=item['task'];d=TASKS[task];q=item['q'];A=item['A']
 if (A/'STRATEGY_INTERRUPTED.json').exists():return False
 try:
  data,path=find_result(R,task,q['run_id'])
  if data is None:return False
 except (json.JSONDecodeError,FileNotFoundError):return False
 if data.get('details'):
  try:
   mode=json.loads((A/'captures/MODE_VERIFIED.json').read_text())
   startup=json.loads((A/'captures/STARTUP_WARMUP_VERIFIED.json').read_text())
   image=json.loads((A/'captures/ALL_FIRST_OBSERVATION.json').read_text())
  except (json.JSONDecodeError,FileNotFoundError):return False
  assert mode['mode']=='PathTracing' and mode['protocol']=='xstar-pt16-rigid-v1' and mode['spp']==16 and mode['denoiser'] is True and mode['warmup_renders']==8 and mode['simulation_device']=='cpu' and mode['use_fabric'] is False
  assert startup['env_physics_groups']==500 and startup['total_physics_steps']>=500 and startup['discarded_rgb_captures']==0
  if task=='play_tic_tac_toe':assert startup['total_physics_steps']==540
  assert image['batch']<=25 and len(image['stats'])==3*image['batch']
  assert all(x['finite'] and x['shape']==[480,640,3] and x['std']>5 and 20<x['mean']<240 for x in image['stats'])
  assert mode['physics_time_before']==mode['physics_time_after']
 before=set(d['rows']);old_excluded=set(d['excluded'])
 collect_attempt(d['rows'],d['excluded'],worker=item['worker'],seed=q['seed'],requested=q['layout_ids'],data=data,log=(A/'client.log').read_text(errors='replace'),attempt=item['attempt'])
 indices={int(v['layout_id']):int(k) for k,v in data.get('details',{}).items()}
 for key in set(d['rows'])-before:d['rows'][key].update(result_source=path,result_index=indices[key[1]],attempt=item['attempt'],client_log=str(A/'client.log'))
 changed=before!=set(d['rows']) or old_excluded!=set(d['excluded'])
 if changed:publish(task)
 return changed

def complete_tasks():
 for task,d in TASKS.items():
  if STATES[task].get('state')=='complete':continue
  expected={(s['seed'],i) for s in PLANS[task]['seeds'] for i in s['layout_ids']}
  assert not set(d['rows'])&set(d['excluded'])
  if set(d['rows'])|set(d['excluded'])!=expected or any(x['task']==task for x in ACTIVE.values()):continue
  per_seed=[]
  for s in PLANS[task]['seeds']:
   selected=[v for v in d['rows'].values() if v['seed']==s['seed']];n=len(selected);assert n>0
   successes=sum(v['success'] for v in selected)
   per_seed.append(dict(seed=s['seed'],valid=n,success=successes,failure=n-successes,unstable=sum(v['seed']==s['seed'] for v in d['excluded'].values()),score=100*sum(v['score'] for v in selected)/n,success_rate=100*successes/n))
  result=dict(task=task,state='complete',planned=len(expected),valid=len(d['rows']),unstable=len(d['excluded']),successes=sum(v['success'] for v in d['rows'].values()),per_seed=per_seed,score=sum(x['score'] for x in per_seed)/len(per_seed),success_rate=sum(x['success_rate'] for x in per_seed)/len(per_seed),details=list(d['rows'].values()),native_exclusions=list(d['excluded'].values()),wall_seconds=time.time()-START,topology='manifest_policy_sim_gpus_three_workers_balanced25_pt16')
  save(G/'tasks'/task/'result.json',result);STATES[task]=result;event('fullsuite_task_complete',task=task,score=result['score'],success_rate=result['success_rate'])

def metric_poll(item):
 global TOTAL_STEPS
 counter=METRICS[item['q']['run_id']];delta=counter.poll();TOTAL_STEPS+=delta;GPU_STEPS[str(item['q']['sim_gpu'])]+=delta

def sample(force=False):
 global LAST_SAMPLE
 now=time.time()
 if not force and now-LAST_SAMPLE<10:return
 LAST_SAMPLE=now
 workers=[]
 for item in ACTIVE.values():
  A=item['A'];mode=A/'captures/MODE_VERIFIED.json';obs=A/'captures/ALL_FIRST_OBSERVATION.json'
  workers.append(dict(pid=item['proc'].pid,task=item['task'],gpu=item['q']['sim_gpu'],slot=item['slot'],seed=item['q']['seed'],layout_ids=item['q']['layout_ids'],run_id=item['q']['run_id'],mode_ready=mode.exists(),first_observation_ready=obs.exists(),metrics=str(A/'metrics.jsonl')))
 v=dict(time=now,epoch=RUN_TAG,model=C.name,started_at=START,env_steps=TOTAL_STEPS,gpu_env_steps=GPU_STEPS,completed_tasks=sum(x.get('state')=='complete' for x in STATES.values()),planned_tasks=54,planned_cases=2700,valid_cases=sum(len(x['rows']) for x in TASKS.values()),native_exclusions=sum(len(x['excluded']) for x in TASKS.values()),active_workers=workers,policy_pid=policy.pid,policy_gpu=M['policy_gpu'],sim_gpus=SIM_GPUS,workers_per_gpu=3)
 save(C/'serial-speed-latest.json',v)
 with (C/'serial-speed-history.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
 progress()

base_env=dict(os.environ,PERF_ROOT=str(C),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES=str(M['policy_gpu']),PYTHONPATH=f'{R}:{OW}/OpenWAM',OW_SERVER_METRICS=str(C/'server-metrics.jsonl'),OW_INPUT_AUDIT=str(C/'input-audit'))
adoption=read_adoption=json.loads((C/'serial-adoption.json').read_text()) if (C/'serial-adoption.json').exists() else None
policy=None

def launch(task,seed,ids,spec):
 global LAST_LAUNCH
 d=TASKS[task]
 assert task not in HOLDS
 for i in ids:assert d['attempts'].get((seed,i),0)<3,f'Unresolved infrastructure case {task}/{seed}/{i}'
 D=G/'tasks'/task;D.mkdir(parents=True,exist_ok=True)
 worker=max([int(p.name.split('-w')[1]) for p in D.glob('seed*-w*') if '-w' in p.name],default=-1)+1
 attempt=max([int(p.name[1:]) for p in D.glob('seed*/a*') if p.name[1:].isdigit()],default=-1)+1
 A=D/f'seed{seed}-w{worker}'/f'a{attempt}';(A/'captures').mkdir(parents=True)
 run_id=f'serial{RUN_TAG}-{task}-s{seed}-w{worker}-a{attempt}'
 suffix=f'job{M["job"]}-serial-{C.name}-gpu{spec["gpu"]}-slot{spec["slot"]}'
 template=json.loads((C.parent/'parallel-cache-template.json').read_text())
 import shutil
 for source,pattern in template['entries']:
  target=G/'cache-sim6'/pattern.format(gpu=spec['gpu'],suffix=suffix)
  if not target.exists():shutil.copytree(source,target,symlinks=True,ignore=shutil.ignore_patterns('documents'))
 env=dict(base_env);env.pop('CUDA_VISIBLE_DEVICES',None)
 env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',ROBODOJO_SAVE_EVAL_VIDEOS='false',ROBODOJO_BATCH_IK='1',EVAL_NUM=str(len(ids)),ROBODOJO_RUN_ID=run_id,ROBODOJO_CACHE_SUFFIX=suffix,ROBODOJO_SIM_DEVICE='cpu',ROBODOJO_MAX_BASH_RETRIES='0',OW_SEED=str(seed),OW_WORKER=str(worker),OW_LAYOUT_IDS=json.dumps(ids),OW_METRICS=str(A/'metrics.jsonl'),RENDER_PROBE_OUT=str(A/'captures'))
 args=['bash',G/'sim-python6-parallel','-u','src/eval_client/main.py','--task_name',task,'--num_envs',str(len(ids)),'--env_cfg_type','arx_x5','--device_id',str(spec['gpu']),'--policy_name','OpenWAM','--host','127.0.0.1','--port',str(PORT),'--protocol','ws','--additional_info',f'ckpt_name={M["checkpoint_label"]},action_type=ee','--seed',str(seed),'--enable_cameras','--headless']
 proc=spawn(args,env,R,A/'client.log',spec['cpus'],f'{task}/seed{seed}/w{worker}/a{attempt}')
 q=dict(task=task,seed=seed,layout_ids=ids,run_id=run_id,pid=proc.pid,sim_gpu=spec['gpu'],slot=spec['slot'],cpus=spec['cpus'],cache_suffix=suffix,topology='manifest_policy_sim_gpus_three_workers_balanced25_pt16')
 save(A/'request.json',q);ACTIVE[spec['slot']]=dict(proc=proc,A=A,q=q,worker=worker,attempt=attempt,began=time.time(),task=task,slot=spec['slot'])
 METRICS[run_id]=MetricSteps(A/'metrics.jsonl')
 for i in ids:d['attempts'][(seed,i)]=d['attempts'].get((seed,i),0)+1
 publish(task);LAST_LAUNCH=time.time();event('serial_worker_started',task=task,gpu=spec['gpu'],slot=spec['slot'],pid=proc.pid,requested=ids)

try:
 for task in priority:
  for retry in range(5):
   try:
    rows,excluded,finished=recover_cases(task);break
   except json.JSONDecodeError:
    if retry==4:raise
    time.sleep(0.5)
  attempts={}
  for p in (G/'tasks'/task).glob('seed*/a*/request.json'):
   if (p.parent/'STRATEGY_INTERRUPTED.json').exists():continue
   q=json.loads(p.read_text())
   for i in q['layout_ids']:attempts[(q['seed'],i)]=attempts.get((q['seed'],i),0)+1
  TASKS[task]=dict(rows=rows,excluded=excluded,attempts=attempts)
  STATES[task]=finished if finished is not None else dict(state='queued',valid=len(rows),unstable=len(excluded),successes=sum(v['success'] for v in rows.values()))
 if adoption:
  cookie=adoption['policy'];policy=ExistingProcess(cookie);CHILDREN.append((policy,cookie));event('policy_adopted',pid=policy.pid)
 else:
  args=[M['policy_python'],'-u',G/'server_entry.py','--config_path',OW/'deploy.yml','--overrides',f'port={PORT}','host=127.0.0.1','bench_name=RoboDojo','task_name=put_bottles_into_dustbin',f'ckpt_name={M["checkpoint_label"]}','env_cfg_type=arx_x5','action_type=ee','seed=1','policy_name=OpenWAM','gpu_id=0',f'openwam_root={OW}/OpenWAM',f'ckpt_dir={C}/foundation-deploy',f'openwam_deploy_config={C}/deploy.yaml','replan_steps=32','allow_dummy_policy=false']
  policy=spawn(args,base_env,R,C/'server-fullsuite.log',M['policy_cpus'],'policy')
 # Native clients connect after scene setup; model and scenes start together.
 if adoption:
  for item in adoption.get('simulators',[]):
   A=pathlib.Path(item['request']).parent;q=json.loads((A/'request.json').read_text());task=A.parents[1].name
   assert task in PLANS and q['sim_gpu'] in SIM_GPUS
   try:proc=ExistingProcess(item['cookie'])
   except RuntimeError:continue
   slot=item['slot'];assert SLOTS[slot]['gpu']==q['sim_gpu'] and slot not in ACTIVE
   worker=int(A.parent.name.split('-w')[1]);attempt=int(A.name[1:])
   ACTIVE[slot]=dict(proc=proc,A=A,q=q,worker=worker,attempt=attempt,began=time.time(),task=task,slot=slot)
   CHILDREN.append((proc,item['cookie']));counter=MetricSteps(A/'metrics.jsonl');counter.poll();METRICS[q['run_id']]=counter
   publish(task);event('simulator_adopted',pid=proc.pid,task=task,slot=slot)
 if adoption:
  # Recompute complete pre-handoff native counters, then only poll future deltas.
  for mp in G.glob('tasks/*/seed*/a*/metrics.jsonl'):
   rq=json.loads((mp.parent/'request.json').read_text());mc=METRICS.get(rq['run_id'],MetricSteps(mp));mc.poll();TOTAL_STEPS+=mc.total;GPU_STEPS[str(rq['sim_gpu'])]+=mc.total
  save(C/'BALANCED_HANDOFF_ADOPTED.json',dict(time=time.time(),controller_pid=os.getpid(),policy_pid=policy.pid,simulator_pids=[x['proc'].pid for x in ACTIVE.values()],slots=SLOTS,preserved_steps=TOTAL_STEPS))
 save(C/'children.json',[q for _,q in CHILDREN]);save(C/'PARALLEL_STARTUP_DISPATCHED.json',{'time':time.time(),'policy_pid':policy.pid,'simulator_pids':[x['proc'].pid for x in ACTIVE.values()]})
 sample(True)
 while sum(s.get('state')=='complete' for s in STATES.values())<54 or ACTIVE:
  if STOP.is_set() or policy.poll() is not None:raise RuntimeError('Controller stopped or policy exited')
  for slot,item in list(ACTIVE.items()):
   metric_poll(item)
   proc=item['proc'];A=item['A']
   log=(A/'client.log').read_text(errors='replace')
   oom=any(x in log for x in ['cudaErrorMemoryAllocation','ERROR_OUT_OF_DEVICE_MEMORY','Out of GPU memory allocating resource'])
   if oom and not (A/'STRATEGY_INTERRUPTED.json').exists():
    save(A/'STRATEGY_INTERRUPTED.json',dict(time=time.time(),reason='GPU memory allocation failure; no policy outcome accepted'))
    event('memory_allocation_failure_quarantined',pid=proc.pid,task=item['task'])
    BATCHES[item['task']]=max(1,(len(item['q']['layout_ids'])+1)//2)
    save(C/'ADAPTIVE_BATCHES.json',BATCHES)
    terminate(proc)
   if not oom:harvest(item)
   if proc.poll() is None:
    fatal=any(x in log[-18000:] for x in ['[Fatal] [carb.crashreporter-breakpad.plugin]','Simulation cannot continue! Error code 700','GPU crash is detected','ERROR_DEVICE_LOST'])
    latest=(A/'metrics.jsonl').stat().st_mtime if (A/'metrics.jsonl').exists() else item['began']
    if not fatal and time.time()-latest<3600:continue
    if fatal:
     FAULTED_GPUS.add(item['q']['sim_gpu']);save(C/'FAULTED_SIM_GPUS.json',dict(time=time.time(),gpus=sorted(FAULTED_GPUS),task=item['task']))
    event('serial_worker_infrastructure_failure',pid=proc.pid,task=item['task'],fatal=fatal);terminate(proc)
   metric_poll(item);harvest(item);del ACTIVE[slot];publish(item['task'])
  complete_tasks()
  # Infrastructure incompatibilities remain pending, never counted as policy failures.
  for t,d in TASKS.items():
   expected={(p['seed'],i) for p in PLANS[t]['seeds'] for i in p['layout_ids']}
   unresolved=expected-set(d['rows'])-set(d['excluded'])
   if any(d['attempts'].get(k,0)>=3 for k in unresolved):HOLDS.setdefault(t,'Infrastructure retry bound reached; raw cases remain pending')
  for t,reason in HOLDS.items():
   if STATES[t].get('state')!='complete':STATES[t].update(state='pending_infrastructure',reason=reason)
  save(C/'INFRASTRUCTURE_HOLDS.json',dict(time=time.time(),tasks=HOLDS))
  if not ACTIVE and FAULTED_GPUS:
   raise RuntimeError('Native simulator GPU failed; dispatch held for recovery, results preserved')
  if not ACTIVE and all(STATES[t].get('state')=='complete' or t in HOLDS for t in PLANS):
   save(C/'FULLSUITE_PENDING_INFRASTRUCTURE.json',dict(time=time.time(),complete=False,completed=sum(x.get('state')=='complete' for x in STATES.values()),pending_tasks=HOLDS));sample(True);break
  if time.time()-LAST_LAUNCH>=3:
   for spec in SLOTS:
    if spec['gpu'] in FAULTED_GPUS:continue
    if spec['slot'] in ACTIVE:continue
    same_gpu=[x for x in ACTIVE.values() if x['q']['sim_gpu']==spec['gpu']]
    q=subprocess.run(['nvidia-smi','-i',str(spec['gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if int(q.stdout.strip())<20000:continue
    # Reserve env capacity before scene allocations become visible in nvidia-smi.
    if sum(len(x['q']['layout_ids']) for x in same_gpu)>=75:continue
    if sum(not (x['A']/'captures/ALL_FIRST_OBSERVATION.json').exists() for x in same_gpu)>=2:continue
    confirmed={t:set(d['rows'])|set(d['excluded']) for t,d in TASKS.items()}
    requests=[dict(task=x['task'],seed=x['q']['seed'],layout_ids=x['q']['layout_ids']) for x in ACTIVE.values()]
    assignment=next_assignment(PLANS,confirmed,requests,[t for t in priority if t not in HOLDS],BATCHES)
    if assignment is None:break
    launch(*assignment,spec);break
  sample();time.sleep(2)
 metrics=[json.loads(x) for x in (C/'server-metrics.jsonl').read_text().splitlines()] if (C/'server-metrics.jsonl').exists() else []
 assert all(x['chunk']==32 for x in metrics if x.get('method')=='get_action_batch')
 save(C/'RUNTIME_VERIFIED.json',{'time':time.time(),'job':M['job'],'chunk':32,'policy_gpu':M['policy_gpu'],'sim_gpus':SIM_GPUS,'workers_per_gpu':3,'renderer':'PathTracing','spp':16,'defer_usd_writeback':True,'capture_eval_warmup_observations':False})
 if sum(x.get('state')=='complete' for x in STATES.values())==54:
  save(C/'FULLSUITE_FINISHED.json',{'t':time.time(),'complete':54,'planned_tasks':54,'scope':'All requested cases; independent native audit required'})
 sample(True)
except BaseException:
 save(C/'FULLSUITE_CONTROLLER_ERROR.json',{'t':time.time(),'error':traceback.format_exc()});raise
finally:
 STOP.set()
 for proc,_ in reversed(CHILDREN):terminate(proc)
