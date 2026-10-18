"""Connect after scene preparation, when the first policy RPC is made."""

from threading import Lock


class DeferredPolicyClient:
    """Preserve the client API while overlapping scene reset with server loading.

    Constructing a WebSocket client may wait for a large policy to load. Scene
    preparation does not need that connection. The first call still waits for
    the normal client handshake, and RPC ordering and errors remain unchanged.
    """

    def __init__(self, factory, **kwargs):
        self._factory = factory
        self._kwargs = dict(kwargs)
        self._instance = None
        self._closed = False
        self._init_lock = Lock()

    def _get(self):
        with self._init_lock:
            if self._closed:
                raise RuntimeError("Policy client is closed")
            if self._instance is None:
                self._instance = self._factory(**self._kwargs)
            return self._instance

    def call(self, *args, **kwargs):
        return self._get().call(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in ("trial_id", "action_case_id", "repeat_index") and self._instance is None:
            return self._kwargs.get(name)
        return getattr(self._get(), name)

    def close(self):
        with self._init_lock:
            if self._closed:
                return
            self._closed = True
            instance = self._instance
        if instance is not None:
            instance.close()
