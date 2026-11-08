"""Paired TIC regression; preserve native scoring and physical execution."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
from measure_native import instrument_factory as native

def instrument_factory(factory):
    base=native(factory)
    def create(*a,**kw):
        env=base(*a,**kw)
        optimized=True
        out=Path(os.environ['RENDER_PROBE_OUT'])
        os.environ.setdefault('PERF_WORKER_OUT',str(out))
        env._stream_vision=lambda *args,**kwargs:None
        if optimized:
            from env.environment.task_env import TaskEnv
            from gripper_snapshot import install
            install(env,enabled=getattr(type(env).__mro__[1],'step',None) is TaskEnv.step,shadow_epochs=64)
            import env.camera_manager.capture.render_sync as rs
            old=rs.wait_for_latest_cameras
            def fence(ctx,*args,**kwargs):
                requested=kwargs.get('min_passes',1)
                if requested!=2:return old(ctx,*args,**kwargs)
                from counter_delivery import capture_fence
                return capture_fence(ctx,env.capture_manager,old)
            rs.wait_for_latest_cameras=fence
        original=env.get_obs_batch
        sequence=[0]
        def observation(*args,**kwargs):
            obs=original(*args,**kwargs)
            if sequence[0]<1:
                rows=[]
                for item in obs:
                    images={}
                    for camera in ['cam_head','cam_left_wrist','cam_right_wrist']:
                        pixels=np.asarray(item['vision'][camera]['color'])
                        images[camera]=dict(shape=list(pixels.shape),sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
                    rows.append(dict(env=int(item['env_idx']),images=images))
                (out/f'INPUT_HASHES_{sequence[0]}.json').write_text(json.dumps(rows,indent=2))
            sequence[0]+=1
            return obs
        env.get_obs_batch=observation
        old_run=env.run_eval
        def run(*args,**kwargs):
            try:return old_run(*args,**kwargs)
            finally:
                (out/'PAIR_RUNTIME_FINAL.json').write_text(json.dumps(dict(time=time.time(),optimized=optimized,observations=sequence[0],gripper_snapshot=getattr(env,'_gripper_snapshot_stats',None),counts=list(env.take_action_cnt)),indent=2))
        env.run_eval=run
        return env
    return create
