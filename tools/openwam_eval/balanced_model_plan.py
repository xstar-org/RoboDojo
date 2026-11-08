"""Allocate exact pending seed/layout keys across nine simulation slots."""
from collections import Counter
import math
from multigpu_plan import pending_keys

def balanced_chunks(plan,max_envs=25):
    assert 1<=max_envs<=25
    chunks=[]
    for part in plan['seeds']:
        ids=part['layout_ids'];groups=math.ceil(len(ids)/max_envs)
        if not groups:continue
        base,extra=divmod(len(ids),groups);offset=0
        for i in range(groups):
            size=base+(i<extra);chunks.append((part['seed'],ids[offset:offset+size]));offset+=size
        assert offset==len(ids)
    return chunks

def next_assignment(plans,confirmed,active,priority,batch=25):
    counts=Counter(q['task'] for q in active);candidates=[]
    for rank,task in enumerate(priority):
        pending=pending_keys(plans[task],confirmed[task],[q for q in active if q['task']==task])
        if pending:candidates.append((counts[task],rank,task,pending))
    if not candidates:return None
    _,_,task,pending=min(candidates)
    maximum=batch[task] if isinstance(batch,dict) else batch
    for seed,ids in balanced_chunks(plans[task],maximum):
        missing=[i for i in ids if (seed,i) in pending]
        if missing:return task,seed,missing
    raise AssertionError('Pending case missing from frozen plan')

def balanced_slots(pools,gpus=None,workers_per_gpu=3):
    gpus=list(gpus) if gpus is not None else list(map(int,pools))
    assert gpus and len(set(gpus))==len(gpus) and workers_per_gpu>0
    return [dict(slot=i,gpu=g,cpus=pools[str(g)]) for i,g in enumerate(gpus*workers_per_gpu)]
