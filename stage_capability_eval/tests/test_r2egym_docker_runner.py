import unittest

from stage_capability_eval.r2egym_docker_runner import DockerCommandResult, safe_relative_path


class DockerRunnerTests(unittest.TestCase):
    def test_safe_relative_paths_and_rejects_escape(self):
        self.assertEqual(safe_relative_path("generated/test_repro.py"), "generated/test_repro.py")
        for value in ("", "/tmp/test.py", "../test.py", "a/../test.py", "a\\test.py", ".git/config"):
            with self.assertRaises(ValueError):
                safe_relative_path(value)

    def test_command_result_persists_only_log_hashes(self):
        result = DockerCommandResult(1, stdout="out", stderr="err")
        encoded = result.to_dict()
        self.assertNotIn("stdout", encoded)
        self.assertNotIn("stderr", encoded)
        self.assertEqual(len(encoded["stdout_sha256"]), 64)
        self.assertEqual(len(encoded["stderr_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
