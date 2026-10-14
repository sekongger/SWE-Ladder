import tempfile
import unittest
from pathlib import Path

from stage_capability_eval.patch import (
    PatchRunnerResult,
    build_public_package,
    evaluate_patch,
)


class FakeRunner:
    def __init__(self, *, apply=True, f2p=(0,), p2p=(0,)):
        self.apply = apply
        self.f2p = list(f2p)
        self.p2p = list(p2p)
        self.calls = []

    def apply_patch(self, workspace, patch, timeout_seconds):
        self.calls.append(("apply", str(workspace), patch))
        return PatchRunnerResult(returncode=0 if self.apply else 1, stderr="bad patch" if not self.apply else "")

    def run_test(self, workspace, test_name, timeout_seconds):
        self.calls.append(("test", str(workspace), test_name))
        values = self.f2p if test_name.startswith("f2p") else self.p2p
        value = values.pop(0) if values else 0
        return value if isinstance(value, PatchRunnerResult) else PatchRunnerResult(returncode=value)


class PatchTests(unittest.TestCase):
    def test_public_package_has_loc_but_no_private_patch_or_verifier(self):
        manifest = [{
            "instance_id": "repo__one", "repo": "repo/one", "base_commit": "abc",
            "problem_statement": "issue", "source_patch_files": ["pkg/a.py"],
            "FAIL_TO_PASS": ["secret"], "patch": "diff --git a/x b/x",
        }]
        loc = [{"instance_id": "repo__one", "files": [{"path": "pkg/a.py", "functions": ["f"]}]}]
        package = build_public_package(manifest, loc)
        self.assertEqual(package[0]["stage"], "PATCH_GIVEN_LOC")
        self.assertEqual(package[0]["oracle_loc"]["files"][0]["path"], loc[0]["files"][0]["path"])
        self.assertEqual(package[0]["oracle_loc"]["files"][0]["functions"], loc[0]["files"][0]["functions"])
        self.assertNotIn("patch_sha256", package[0]["oracle_loc"])
        self.assertNotIn("patch", package[0])
        self.assertNotIn("FAIL_TO_PASS", package[0])
        self.assertNotIn("source_patch_files", package[0])

    def test_conditional_resolved_requires_f2p_and_no_p2p_regression(self):
        task = {"instance_id": "one", "oracle_loc": {"files": [{"path": "a.py"}]}}
        verifier = {"FAIL_TO_PASS": ["f2p-test"], "PASS_TO_PASS": ["p2p-test"]}
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_patch(task, "diff --git a/a.py b/a.py\n", verifier,
                                    workspace=Path(directory), runner=FakeRunner())
        self.assertEqual(result["status"], "conditional_resolved")
        self.assertTrue(result["patch_applied"])
        self.assertEqual(result["f2p"]["passed"], 1)
        self.assertEqual(result["p2p"]["regressions"], 0)

        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_patch(task, "diff --git a/a.py b/a.py\n", verifier,
                                    workspace=Path(directory), runner=FakeRunner(p2p=(1,)))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_category"], "p2p_regression")

    def test_invalid_timeout_infra_and_dry_run_are_explicit(self):
        task = {"instance_id": "one"}
        with tempfile.TemporaryDirectory() as directory:
            bad = evaluate_patch(task, "not a diff", {}, workspace=directory, runner=FakeRunner())
            timed = evaluate_patch(task, "diff --git a/a b/b\n", {"FAIL_TO_PASS": ["f2p"]},
                                   workspace=directory, runner=FakeRunner(f2p=(PatchRunnerResult(None, timed_out=True),)))
            dry = evaluate_patch(task, "diff --git a/a b/b\n", {}, workspace=directory,
                                 runner=None, dry_run=True)
        self.assertEqual(bad["status"], "invalid")
        self.assertEqual(timed["status"], "timeout")
        self.assertEqual(dry["status"], "not_applicable")
        self.assertEqual(dry["failure_category"], "dry_run")

        with tempfile.TemporaryDirectory() as directory:
            infra = evaluate_patch(task, "diff --git a/a b/b\n", {"FAIL_TO_PASS": ["f2p"]},
                                   workspace=directory,
                                   runner=FakeRunner(f2p=(PatchRunnerResult(None, infra_error=True),)))
        self.assertEqual(infra["status"], "infra_error")


if __name__ == "__main__":
    unittest.main()
