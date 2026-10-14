import unittest
import importlib.util

from stage_capability_eval.execution import build_test_spec, parse_junit


class ExecutionTests(unittest.TestCase):
    def test_junit_parser_preserves_case_status(self):
        xml = '<testsuite><testcase classname="m" name="ok"/><testcase classname="m" name="bad"><failure/></testcase><testcase classname="m" name="skip"><skipped/></testcase></testsuite>'
        self.assertEqual(parse_junit(xml), {'m::ok': 'passed', 'm::bad': 'failed', 'm::skip': 'skipped'})

    @unittest.skipUnless(importlib.util.find_spec('swebench'), 'swebench harness is installed in the Docker adapter environment')
    def test_reference_script_reuses_image_dependencies(self):
        row = {
            'instance_id': 'astropy__astropy-12907', 'repo': 'astropy/astropy', 'version': '4.3',
            'base_commit': 'd16bfe05a744909de4b27f5875fe0d4ed41ce607',
            'test_patch': 'diff --git a/test.py b/test.py\n', 'FAIL_TO_PASS': [], 'PASS_TO_PASS': [],
        }
        spec, skipped = build_test_spec(row, reuse_image_dependencies=True)
        self.assertIsNotNone(skipped)
        self.assertNotIn('pip install -e', spec.eval_script)


if __name__ == '__main__': unittest.main()
