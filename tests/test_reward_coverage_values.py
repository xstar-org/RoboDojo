"""Coverage cache must preserve strict geometry thresholds and per-env identity."""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(os.environ.get('XSTAR_REWARD_TEST_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from utils.transformer import calc_polygon, check_1d


def load(path, names, namespace):
    nodes = [m for c in ast.parse((ROOT/path).read_text()).body if isinstance(c, ast.ClassDef)
             for m in c.body if isinstance(m, ast.FunctionDef) and m.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
    return namespace


class CoverageValueTests(unittest.TestCase):
    def setUp(self):
        ns = dict(np=np, calc_polygon=calc_polygon, check_1d=check_1d)
        load('env/scene_manager/layout_manager.py', {'get_instance_projected_bbox'}, ns)
        load('env/reward_manager/func_parser.py', {'is_A_cover_B', '_cached_A_cover_B'}, ns)
        self.position = {(i, n): np.zeros(3) for i in range(2) for n in ('a','b')}
        vertices = {n: np.array([[x,y,z] for x in (-r,r) for y in (-r,r) for z in (-r,r)])
                    for n,r in [('a',1.),('b',.5)]}
        m = SimpleNamespace(_projected_bbox_cache=None,
            get_instance_name=lambda env_idx,label: label,
            get_instance_pose=lambda env_idx,inst_name: (self.position[env_idx,inst_name], np.array([1.,0.,0.,0.])),
            get_instance_bbox_vertices=lambda inst_name,env_idx: vertices[inst_name])
        m.get_instance_projected_bbox = lambda name,i: ns['get_instance_projected_bbox'](m,name,i)
        self.manager = m
        parser = SimpleNamespace(layout_manager=m, num_envs=2)
        parser._cached_A_cover_B = lambda args: ns['_cached_A_cover_B'](parser,args)
        self.evaluate = lambda i: ns['is_A_cover_B'](parser,dict(env_idx=i,label_A='a',label_B='b'))

    def test_thresholds_and_environment_cache_keys(self):
        # B top must be strictly below A top; B bottom strictly above A bottom - .03.
        for axis, edge in ((0,.5),(2,.5),(2,-.53)):
            for offset in (-1e-7,0.,1e-7):
                self.position[0,'b'][:] = 0.
                self.position[0,'b'][axis] = edge + offset
                self.position[1,'b'][:] = [5.,0.,0.]
                self.manager._projected_bbox_cache = None
                expected = [self.evaluate(i) for i in range(2)]
                self.manager._projected_bbox_cache = {}
                for _ in range(3):
                    self.assertEqual([self.evaluate(i) for i in range(2)], expected)
                self.assertEqual(len(self.manager._projected_bbox_cache), 4)

    def test_outside_scope_reads_new_geometry(self):
        self.assertEqual(self.evaluate(0),1.)
        self.position[0,'b'][0] = 5.
        self.assertEqual(self.evaluate(0),0.)
        self.assertIsNone(self.manager._projected_bbox_cache)


if __name__ == '__main__':
    unittest.main()
