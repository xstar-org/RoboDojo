import os,pathlib,sys,runpy,time,json,contextvars
C=pathlib.Path(__file__).resolve().parent
R=C/'RoboDojo'; X=R/'XPolicyLab'
sys.path[:0]=[str(X),str(R)]

# Private deployment repair: OrderedDict.popitem can invoke the overridden
# __getitem__, whose move_to_end invalidates the eviction operation.
from collections import OrderedDict
from openwam.deploy import engine as _cache_engine
def _safe_prompt_cache_set(self,key,value):
    if key in self:self.move_to_end(key)
    OrderedDict.__setitem__(self,key,value)
    while len(self)>self._maxsize:
        evicted=next(iter(self))
        OrderedDict.__delitem__(self,evicted)
        if not self._evict_warned:
            self._evict_warned=True
            _cache_engine.logger.warning('prompt cache safe eviction active, maxsize=%d',self._maxsize)
_cache_engine._BoundedPromptEmbedCache.__setitem__=_safe_prompt_cache_set
_test_cache=_cache_engine._BoundedPromptEmbedCache(32)
for _i in range(768):_test_cache[str(_i)]=_i
assert len(_test_cache)==32 and list(_test_cache)==list(map(str,range(736,768)))
assert _test_cache['736']==736
_test_cache['768']=768
assert '736' in _test_cache and '737' not in _test_cache and len(_test_cache)==32
print('[deployment repair] bounded prompt cache eviction and LRU checks passed',flush=True)

from client_server.ws.model_server import PolicyServer
identity=contextvars.ContextVar('worker_identity',default='unknown')
dispatch=PolicyServer._dispatch_frame
async def scoped_dispatch(self,frame):
    token=identity.set(frame.evaluation_id)
    try:return await dispatch(self,frame)
    finally:identity.reset(token)
async def scoped_call(self,method,*args):
    import profile_policy
    # Full queue has no four-client barrier; tail batches may be smaller.
    profile_policy.ACTIVE.set(identity.get())
    submitted=time.perf_counter()
    async with self._model_lock:
        started=time.perf_counter(); key=identity.get()
        import profile_policy; profile_policy.install(self.model)
        states=self.__dict__.setdefault('_openwam_worker_states',{})
        batch,order=states.get(key,({},[]))
        self.model._batch=batch; self.model._order=order
        batch_ids=list(args[0]) if method.__name__=='get_action_batch' and args else []
        try:
            result=await self._invoke_method(method,*args)
            if method.__name__=='get_action_batch':
                assert len(result)==len(batch_ids)
                record=dict(worker=key,method=method.__name__,batch=len(result),chunk=len(result[0]) if result else 0,seconds=time.perf_counter()-started,queue_seconds=started-submitted,t=time.time())
                with open(os.environ['OW_SERVER_METRICS'],'a') as f:f.write(json.dumps(record)+'\n')
            return result
        finally:states[key]=(self.model._batch,self.model._order)
PolicyServer._dispatch_frame=scoped_dispatch
PolicyServer._call_model_method=scoped_call
sys.argv[0]=str(X/'setup_policy_server.py')
runpy.run_path(sys.argv[0],run_name='__main__')
