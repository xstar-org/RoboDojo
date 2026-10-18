import pathlib,tempfile,types,unittest
import importlib.util
_spec=importlib.util.spec_from_file_location('parallel_ready_client', pathlib.Path(__file__).resolve().parents[1]/'scripts/internal/parallel_ready_client.py')
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
install_client_gate=_module.install_client_gate

class StartupOrdering(unittest.TestCase):
 def test_scene_before_ready_connection_and_rpc_unchanged(self):
  events=[]
  class Client:
   def __init__(self,**kwargs):events.append(('connect',kwargs))
   def call(self,*a,**kw):events.append(('rpc',a,kw));return 'ok'
   def close(self):events.append(('close',))
  with tempfile.TemporaryDirectory() as d:
   ready=pathlib.Path(d)/'SERVER_READY';m=types.SimpleNamespace(WsModelClient=Client)
   install_client_gate(m,ready,poll_s=.001)
   client=m.WsModelClient(trial_id='trial');self.assertEqual(client.trial_id,'trial')
   self.assertEqual(events,[])
   events.append(('scene_initialized',));ready.write_text('model-loaded')
   payload={'env_idx_list':[0,1]}
   self.assertEqual(client.call(func_name='reset'),'ok')
   client.call(func_name='get_action_batch',obs=payload);client.close();client.close()
   self.assertEqual([e[0]for e in events],['scene_initialized','connect','rpc','rpc','close'])
   self.assertIs(events[3][2]['obs'],payload)
 def test_timeout_does_not_connect(self):
  with tempfile.TemporaryDirectory() as d:
   def bad(**kwargs):raise AssertionError('must wait for model READY')
   m=types.SimpleNamespace(WsModelClient=bad);install_client_gate(m,pathlib.Path(d)/'READY',timeout_s=0)
   c=m.WsModelClient()
   with self.assertRaises(TimeoutError):c.call('reset')
   c.close()
 def test_unused_client_does_not_connect(self):
  def bad(**kwargs):raise AssertionError('unused client connected')
  m=types.SimpleNamespace(WsModelClient=bad);install_client_gate(m,'unused-ready')
  c=m.WsModelClient();c.close();c.close()
  with self.assertRaises(RuntimeError):c.call('reset')
 def test_connection_error_propagates(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'READY';p.touch()
   def bad(**kwargs):raise ValueError('handshake failed')
   m=types.SimpleNamespace(WsModelClient=bad);install_client_gate(m,p)
   c=m.WsModelClient()
   with self.assertRaisesRegex(ValueError,'handshake failed'):c.call('reset')

if __name__=='__main__':unittest.main()
