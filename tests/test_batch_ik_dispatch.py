import ast,os,unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
import numpy as np
R=Path(os.environ.get('CODEX_IK_REPO',str(Path(__file__).resolve().parents[1])))
node=next(n for n in ast.walk(ast.parse((R/'src/eval_client/eval_env.py').read_text(encoding='utf-8')))if isinstance(n,ast.FunctionDef)and n.name=='take_action_batch')
ns=dict(os=os,np=np,deepcopy=deepcopy)
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_frozen_take_action_batch','exec'),ns)
run=ns['take_action_batch']
manager_node=next(n for n in ast.walk(ast.parse((R/'env/robot_manager/robot_manager.py').read_text(encoding='utf-8')))if isinstance(n,ast.FunctionDef)and n.name=='solve_ik_batch')
exec(compile(ast.Module(body=[manager_node],type_ignores=[]),'actual_solve_ik_batch','exec'),ns)
manager_batch=ns['solve_ik_batch']
def fixture(n=3):
    calls=[];controls={}
    robots=[NS(type='target',robot_name='x5',arm_name=side+'_arm',gripper_name=side+'_ee',ee_type='gripper',gripper_move=dict(sign=1,mimic=[0,1,0]),gripper_scale=[0,1])for side in ('left','right')]
    def solve(poses,ids,robot):
        calls.append((robot.arm_name,list(ids)));return [dict(status='Success',joint_value=np.full(6,p[0]))for p in poses]
    def scalar(target_pose,env_idx,robot):
        calls.append(('scalar',robot.arm_name,env_idx));return dict(status='Success',joint_value=np.full(6,target_pose[0]))
    def process(info,i):controls[i]=info;return[info]
    manager=NS(robot_list=robots,solve_ik_batch=solve,solve_ik=scalar,process_name=lambda name:name+'_joint_state',control_manager=NS(push=lambda*a:None))
    if n == 1:
        manager.num_envs=n
        manager.solve_ik_batch=MethodType(manager_batch,manager)
    env=NS(physx_monitor_enabled=False,num_envs=n,robot_manager=manager,take_action_cnt=[0]*n,step_lim=800,end_flag=[False]*n,validate_action_dict=lambda a:None,get_action_type=lambda a:'joint'if'left_arm_joint_state'in a else'ee',process_control_info=process,have_empty=lambda ids:True,reward_manager=NS(step=lambda**kw:None),is_episode_end=lambda:None)
    actions=[dict(left_ee_pose=[i+1]*7,right_ee_pose=[i+101]*7,left_ee_joint_state=[.5],right_ee_joint_state=[.5])for i in range(n)]
    return env,actions,calls,controls
class Dispatch(unittest.TestCase):
    def test_dual_arms_same_robot_name_and_32_envs(self):
        e,a,c,o=fixture(32)
        run(e,a)
        self.assertEqual(c,[('left_arm',list(range(32))),('right_arm',list(range(32)))])
        for i in range(32):
            np.testing.assert_array_equal(o[i]['left_arm_joint_state']['position'],np.full(6,i+1))
            np.testing.assert_array_equal(o[i]['right_arm_joint_state']['position'],np.full(6,i+101))
        self.assertEqual(e.take_action_cnt,[1]*32)
    def test_permuted_environment_ids(self):
        e,a,c,o=fixture()
        run(e,[a[2],a[0]],[2,0])
        self.assertEqual(c,[('left_arm',[2,0]),('right_arm',[2,0])]);self.assertEqual(e.take_action_cnt,[1,0,1])
        self.assertEqual(o[2]['left_arm_joint_state']['position'][0],3)
    def test_finished_and_limit_envs_skip(self):
        e,a,c,o=fixture();e.end_flag[1]=True;e.take_action_cnt[2]=800
        run(e,a)
        self.assertEqual(c,[('left_arm',[0]),('right_arm',[0])]);self.assertEqual(set(o),{0})
    def test_tail_one_and_empty(self):
        e,a,c,o=fixture()
        run(e,[a[1]],[1]);run(e,[],[])
        self.assertEqual(c,[('left_arm',[1]),('right_arm',[1])]);self.assertEqual(set(o),{1})
    def test_single_env_retains_scalar(self):
        e,a,c,o=fixture(1)
        run(e,a)
        self.assertTrue(all(r[0]=='scalar'for r in c))
    def test_invalid_result_length_fails_before_counter_change(self):
        e,a,c,o=fixture();e.robot_manager.solve_ik_batch=lambda*a:[]
        with self.assertRaises(RuntimeError):run(e,a)
        self.assertEqual(e.take_action_cnt,[0]*3)
    def test_duplicate_ids_fail(self):
        e,a,c,o=fixture()
        with self.assertRaises(ValueError):run(e,a[:2],[1,1])
        self.assertEqual(c,[])
    def test_joint_action_does_not_call_ik(self):
        e,a,c,o=fixture()
        for i,row in enumerate(a):row.update(left_arm_joint_state=[i]*6,right_arm_joint_state=[100+i]*6)
        run(e,a)
        self.assertEqual(c,[]);self.assertEqual(e.take_action_cnt,[1]*3)
if __name__=='__main__':unittest.main(verbosity=2)
