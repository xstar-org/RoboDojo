import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1]/'src/eval_client/policy_timeout.py'
spec = importlib.util.spec_from_file_location('policy_timeout', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TimeoutTests(unittest.TestCase):
    @patch.dict('os.environ', {}, clear=True)
    def test_default_preserved_and_yaml_supported(self):
        self.assertIsNone(module.policy_request_timeout({}))
        self.assertEqual(module.policy_request_timeout({'request_timeout_s':300}), 300)

    @patch.dict('os.environ', {'ROBODOJO_POLICY_REQUEST_TIMEOUT_S':'10800'}, clear=True)
    def test_remote_agent_budget_overrides_yaml(self):
        self.assertEqual(module.policy_request_timeout({'request_timeout_s':120}), 10800)

    def test_invalid_budgets_fail_before_connection(self):
        for value in ('0','-1','nan','inf','invalid',''):
            with self.subTest(value=value), patch.dict('os.environ', {'ROBODOJO_POLICY_REQUEST_TIMEOUT_S':value}, clear=True):
                with self.assertRaises(ValueError):
                    module.policy_request_timeout({})
        with patch.dict('os.environ', {}, clear=True), self.assertRaises(ValueError):
            module.policy_request_timeout({'request_timeout_s':True})


if __name__ == '__main__':
    unittest.main()
