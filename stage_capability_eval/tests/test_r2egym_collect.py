import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from stage_capability_eval import r2egym_collect as collect


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(instance_id='org__repo-1', repo='org/repo', version='1',
                        docker_image='swebench:example', base_commit='abc',
                        problem_statement='An example issue', patch='SECRET_PATCH',
                        test_patch='SECRET_TEST', hints_text='SECRET_HINT',
                        oracle_loc={'files': [{'path': 'src.py', 'functions': ['f']}],
                                    'patch_sha256': 'SECRET_HASH'})

    def test_loc_input_has_no_oracle_or_private_fields(self):
        task = collect.build_stage_task(self.row, 'loc')
        text = json.dumps(task)
        self.assertNotIn('SECRET', text)
        self.assertNotIn('src.py', text)
        self.assertNotIn('oracle_loc', task)
        self.assertEqual(json.loads(task['make_test_spec'])['FAIL_TO_PASS'], [])

    def test_given_loc_stages_share_sanitized_oracle(self):
        repro = collect.build_stage_task(self.row, 'repro_given_loc')
        patch = collect.build_stage_task(self.row, 'patch_given_loc')
        self.assertIn('src.py', repro['problem_statement'])
        self.assertIn('src.py', patch['problem_statement'])
        self.assertNotIn('SECRET', json.dumps(repro))
        self.assertIn('pytest', repro['problem_statement'])

    def test_finish_json_and_invalid_output(self):
        value = {'files': ['src.py'], 'functions': [{'file': 'src.py', 'function': 'f'}]}
        action = '<function=finish><parameter=command>submit</parameter><parameter=result>' + json.dumps(value) + '</parameter></function>'
        self.assertEqual(collect.parse_finish_json(action), value)
        self.assertEqual(collect.parse_finish_json('```json\n' + json.dumps(value) + '\n```'), value)
        with self.assertRaises(ValueError):
            collect.parse_finish_json('I fixed it')

    def test_repro_capture_after_agent_stages_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run(command):
                result = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True)
                return result.stdout, str(result.returncode)
            run('git init -q')
            (root / 'src.py').write_text('x = 1\n')
            run('git add src.py')
            run('git -c user.name=test -c user.email=test@example.org commit -qm base')
            base = run('git rev-parse HEAD')[0].strip()
            (root / 'test_repro.py').write_text('def test_issue():\n    assert False\n')
            run('git add -A')
            artifact = collect.capture_repro(run, base, {'files': ['test_repro.py'], 'command': ['python', '-m', 'pytest', 'test_repro.py', '-q']})
            self.assertEqual(artifact['files']['test_repro.py'], (root / 'test_repro.py').read_text())
            (root / 'src.py').write_text('x = 2\n')
            with self.assertRaisesRegex(ValueError, 'tracked'):
                collect.capture_repro(run, base, {'files': ['test_repro.py'], 'command': ['python', '-m', 'pytest', 'test_repro.py']})


if __name__ == '__main__':
    unittest.main()
