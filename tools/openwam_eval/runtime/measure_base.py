import os,json,pathlib,time,functools
def instrument_factory(factory):
    from env.seed_manager.seed_manager import SeedManager
    original=SeedManager.init_eval
    def init(self,*a,**kw):
        original(self,*a,**kw)
        ids=json.loads(os.environ['OW_LAYOUT_IDS'])
        assert all(i in self.seed_info for i in ids)
        ids=[i for i in ids if i in self.seed_list]
        self.seed_list=ids; self.st_idx=0; self.ed_idx=len(ids); self.idx=0
        print('[OpenWAM experiment] layouts',ids,flush=True)
    SeedManager.init_eval=init
    def create(*a,**kw):
        env=factory(*a,**kw)
        out=pathlib.Path(os.environ['OW_METRICS'])
        def record(data):
            data.update(t=time.time(),mono=time.perf_counter())
            with out.open('a') as f:f.write(json.dumps(data)+'\n')
        record(dict(kind='created',num_envs=env.num_envs,tensor_device=str(env.device),use_fabric=bool(env.use_fabric)))
        for name in ['reset','get_obs_batch','take_action_batch','run_eval']:
            old=getattr(env,name)
            def wrap(*args,_old=old,_name=name,**kwargs):
                before=list(env.take_action_cnt); start=time.perf_counter()
                if _name=='run_eval':record(dict(kind='run_start'))
                if _name=='run_eval' and os.environ.get('OW_LAYOUT_DIAGNOSTICS')=='1':
                    record(dict(kind='diagnostic_reset_complete',counts=list(env.take_action_cnt)))
                    print('[layout diagnostic] reset succeeded; policy actions deliberately not requested',flush=True)
                    return
                try:return _old(*args,**kwargs)
                finally:
                    data=dict(kind=_name,seconds=time.perf_counter()-start)
                    if _name=='reset':
                        context=getattr(getattr(env,'sim',None),'sim',None)
                        data.update(tensor_device=str(env.device),use_fabric=bool(env.use_fabric),simulation_device=str(context.device) if context is not None else None)
                        if os.environ.get('OW_LAYOUT_DIAGNOSTICS')=='1':data['layout_ids']=dict(getattr(env,'current_env_seed_map',{}))
                    if _name=='take_action_batch':data.update(steps=sum(max(0,b-a) for a,b in zip(before,env.take_action_cnt)),counts=list(env.take_action_cnt))
                    if _name=='run_eval':data.update(counts=list(env.take_action_cnt),success=env.success_nums,fail=env.fail_nums,score=env.total_score)
                    record(data)
            setattr(env,name,wrap)
        audit_original=env.get_obs_batch
        audit_sequence={}
        def audited_observation(*args,**kwargs):
            observations=audit_original(*args,**kwargs)
            for obs in observations:
                idx=int(obs['env_idx']);layout=int(env.env_seeds[idx]);seq=audit_sequence.get(layout,0);audit_sequence[layout]=seq+1
                obs['_input_audit']=dict(run_id=os.environ['ROBODOJO_RUN_ID'],seed=int(os.environ['OW_SEED']),layout_id=layout,worker=int(os.environ['OW_WORKER']),env_idx=idx,obs_sequence=seq)
            return observations
        env.get_obs_batch=audited_observation
        return env
    return create
