import concurrent.futures
import pytest
from src.eval_client.deferred_policy_client import DeferredPolicyClient


def test_scene_preparation_precedes_handshake_and_rpc_order_is_unchanged():
    events = []

    class Client:
        def __init__(self, **kwargs):
            events.append(('connect', kwargs))

        def call(self, *args, **kwargs):
            events.append(('call', args, kwargs))
            return 'reply'

        def close(self):
            events.append(('close',))

    client = DeferredPolicyClient(Client, trial_id='trial')
    assert client.trial_id == 'trial' and not events
    events.append(('scene_reset_completed',))
    assert client.call(func_name='reset') == 'reply'
    client.call(func_name='get_action_batch', obs={'env_idx_list': [0, 1]})
    client.close()
    client.close()
    assert [event[0] for event in events] == [
        'scene_reset_completed', 'connect', 'call', 'call', 'close'
    ]
    assert events[1][1] == {'trial_id': 'trial'}


def test_unused_client_closes_without_connecting():
    def factory():
        raise AssertionError('unexpected connection')

    client = DeferredPolicyClient(factory)
    client.close()
    with pytest.raises(RuntimeError):
        client.call('reset')


def test_concurrent_first_use_constructs_one_connection():
    events = []

    class Client:
        def __init__(self):
            events.append('connect')

        def call(self, value):
            return value

        def close(self):
            events.append('close')

    client = DeferredPolicyClient(Client)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        assert list(pool.map(client.call, range(8))) == list(range(8))
    client.close()
    assert events == ['connect', 'close']


def test_handshake_failure_reaches_caller():
    def factory():
        raise ValueError('handshake failure')

    client = DeferredPolicyClient(factory)
    with pytest.raises(ValueError, match='handshake'):
        client.call('reset')
    client.close()
