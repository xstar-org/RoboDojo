import contextvars
import json
import os
import pathlib
import runpy
import sys
import time

C = pathlib.Path(__file__).resolve().parent
R = C / "RoboDojo"
X = R / "XPolicyLab"
sys.path[:0] = [str(X), str(R)]

from openwam.deploy import engine
from policy_compat import repair_prompt_cache

repair_prompt_cache(engine)

from client_server.ws.model_server import PolicyServer

identity = contextvars.ContextVar("worker_identity", default="unknown")
dispatch = PolicyServer._dispatch_frame


async def scoped_dispatch(self, frame):
    token = identity.set(frame.evaluation_id)
    try:
        return await dispatch(self, frame)
    finally:
        identity.reset(token)


async def scoped_call(self, method, *args):
    import profile_policy

    # Associate each model call with its own evaluation history.
    profile_policy.ACTIVE.set(identity.get())
    submitted = time.perf_counter()
    async with self._model_lock:
        started = time.perf_counter()
        key = identity.get()
        profile_policy.install(self.model)
        states = self.__dict__.setdefault("_openwam_worker_states", {})
        batch, order = states.get(key, ({}, []))
        self.model._batch = batch
        self.model._order = order
        batch_ids = list(args[0]) if method.__name__ == "get_action_batch" and args else []
        try:
            result = await self._invoke_method(method, *args)
            if method.__name__ == "get_action_batch":
                assert len(result) == len(batch_ids)
                record = dict(
                    worker=key,
                    method=method.__name__,
                    batch=len(result),
                    chunk=len(result[0]) if result else 0,
                    seconds=time.perf_counter() - started,
                    queue_seconds=started - submitted,
                    t=time.time(),
                )
                with open(os.environ["OW_SERVER_METRICS"], "a") as f:
                    f.write(json.dumps(record) + "\n")
            return result
        finally:
            states[key] = (self.model._batch, self.model._order)


PolicyServer._dispatch_frame = scoped_dispatch
PolicyServer._call_model_method = scoped_call
sys.argv[0] = str(X / "setup_policy_server.py")
runpy.run_path(sys.argv[0], run_name="__main__")
