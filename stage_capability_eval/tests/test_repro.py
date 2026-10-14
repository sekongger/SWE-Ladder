import unittest

from stage_capability_eval.repro import (
    ReproEvaluator,
    artifact_sha256,
    build_private_metadata,
    build_public_repro_tasks,
    validate_artifact,
)


class ReproPackagingTests(unittest.TestCase):
    def test_public_package_joins_oracle_and_drops_gold_fields(self):
        manifest = [{
            "instance_id": "repo__x-1",
            "repo": "repo/x",
            "base_commit": "abc",
            "problem_statement": "issue",
            "source_patch_files": ["pkg/a.py"],
            "patch": "SECRET SOURCE PATCH",
            "test_patch": "SECRET TEST PATCH",
            "FAIL_TO_PASS": ["tests/test_x.py::test_x"],
            "PASS_TO_PASS": ["tests/test_y.py::test_y"],
        }]
        oracle = [{"instance_id": "repo__x-1", "files": [{"path": "pkg/a.py", "functions": ["f"]}]}]
        tasks = build_public_repro_tasks(manifest, oracle)
        self.assertEqual(tasks[0]["oracle_loc"]["files"][0]["path"], oracle[0]["files"][0]["path"])
        self.assertEqual(tasks[0]["oracle_loc"]["files"][0]["functions"], oracle[0]["files"][0]["functions"])
        self.assertNotIn("patch_sha256", tasks[0]["oracle_loc"])
        self.assertEqual(tasks[0]["task_type"], "REPRO_GIVEN_LOC")
        for key in ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"):
            self.assertNotIn(key, tasks[0])
        self.assertNotIn("hints_text", tasks[0])
        self.assertNotIn("source_patch_files", tasks[0])

    def test_private_metadata_keeps_only_evaluator_references(self):
        rows = [{"instance_id": "x", "FAIL_TO_PASS": ["a"], "PASS_TO_PASS": ["b"], "test_patch": "diff"}]
        private = build_private_metadata(rows)
        self.assertEqual(private[0]["FAIL_TO_PASS"], ["a"])
        self.assertEqual(private[0]["test_patch"], "diff")


class ReproEvaluatorTests(unittest.TestCase):
    def _artifact(self):
        return {
            "files": {"generated/test_repro.py": "def test_bug(): pass\n"},
            "command": ["pytest", "generated/test_repro.py"],
        }

    def test_valid_requires_buggy_failure_and_fixed_pass(self):
        def runner(task, version, artifact):
            return {
                "status": "completed",
                "tests": {"generated/test_repro.py::test_bug": "failed" if version == "buggy" else "passed"},
                "artifact_sha256": artifact_sha256(artifact),
                "log_path": f"/tmp/{version}.log",
            }

        result = ReproEvaluator(runner).evaluate(
            [{"instance_id": "x"}], {"x": self._artifact()}
        )
        self.assertEqual(result["records"][0]["status"], "completed")
        self.assertTrue(result["records"][0]["valid_reproduction"])
        self.assertEqual(result["aggregate"]["planned_count"], 1)
        self.assertEqual(result["aggregate"]["valid_reproduction_rate"], 1.0)

    def test_top_level_pass_does_not_count_without_per_test_outcomes(self):
        def runner(task, version, artifact):
            return {"status": "completed", "artifact_sha256": artifact_sha256(artifact)}

        result = ReproEvaluator(runner).evaluate([{"instance_id": "x"}], {"x": self._artifact()})
        self.assertFalse(result["records"][0]["valid_reproduction"])
        self.assertEqual(result["records"][0]["failure_category"], "invalid_runner_result")
        self.assertEqual(result["records"][0]["status"], "infra_error")

    def test_same_ids_and_outcomes_are_required(self):
        def runner(task, version, artifact):
            tests = {"t": "failed"} if version == "buggy" else {"other": "passed"}
            return {"status": "completed", "tests": tests, "artifact_sha256": artifact_sha256(artifact)}

        result = ReproEvaluator(runner).evaluate([{"instance_id": "x"}], {"x": self._artifact()})
        self.assertFalse(result["records"][0]["valid_reproduction"])
        self.assertEqual(result["records"][0]["failure_category"], "test_id_mismatch")

    def test_all_runner_outcomes_are_preserved_and_scored(self):
        outcomes = {
            "valid": ("completed", {"t": "failed"}, {"t": "passed"}),
            "skipped": ("completed", {"t": "skipped"}, {"t": "skipped"}),
            "timeout": ("timeout", None, None),
            "infra": ("infra_error", None, None),
            "invalid": ("model_invalid", None, None),
        }

        def runner(task, version, artifact):
            status, buggy, fixed = outcomes[task["instance_id"]]
            result = {"status": status, "artifact_sha256": artifact_sha256(artifact), "log_path": "run.log"}
            if status == "completed":
                result["tests"] = buggy if version == "buggy" else fixed
            return result

        artifacts = {key: self._artifact() for key in outcomes}
        result = ReproEvaluator(runner).evaluate(
            [{"instance_id": key} for key in outcomes], artifacts
        )
        records = {row["instance_id"]: row for row in result["records"]}
        self.assertTrue(records["valid"]["valid_reproduction"])
        self.assertFalse(records["skipped"]["valid_reproduction"])
        self.assertEqual(records["timeout"]["status"], "timeout")
        self.assertEqual(records["infra"]["status"], "infra_error")
        self.assertEqual(records["invalid"]["status"], "model_invalid")
        self.assertEqual(records["timeout"]["buggy_outcome"]["status"], "timeout")
        self.assertIsNone(result["aggregate"]["valid_reproduction_rate"])
        self.assertEqual(result["aggregate"]["planned_count"], 5)
        self.assertEqual(result["aggregate"]["pending_count"], 2)
        self.assertEqual(result["aggregate"]["failure_count"], 4)

    def test_artifact_validation_hash_and_copies(self):
        artifact = self._artifact()
        first = validate_artifact(artifact)
        second = validate_artifact({"command": list(artifact["command"]), "files": dict(artifact["files"])})
        self.assertEqual(artifact_sha256(first), artifact_sha256(second))
        self.assertEqual(artifact, self._artifact())
        for bad in (
            {"files": {"": "x"}, "command": ["pytest"]},
            {"files": {"../escape.py": "x"}, "command": ["pytest"]},
            {"files": {"ok.py": "x"}, "command": []},
        ):
            with self.assertRaises(ValueError):
                validate_artifact(bad)

    def test_runner_cannot_mutate_shared_artifact_and_tampering_is_infra(self):
        seen = []
        original = self._artifact()

        def runner(task, version, artifact):
            seen.append(artifact)
            artifact["files"]["generated/test_repro.py"] = "MUTATED"
            return {
                "status": "completed",
                "tests": {"t": "failed" if version == "buggy" else "passed"},
                "artifact_sha256": artifact_sha256(artifact),
            }

        result = ReproEvaluator(runner).evaluate([{"instance_id": "x"}], {"x": original})
        self.assertEqual(result["records"][0]["status"], "infra_error")
        self.assertEqual(result["records"][0]["failure_category"], "artifact_tampered")
        self.assertEqual(original, self._artifact())
        self.assertIsNot(seen[0], seen[1])

    def test_collection_errors_and_all_skipped_are_never_valid(self):
        def runner(task, version, artifact):
            result = {"status": "completed", "artifact_sha256": artifact_sha256(artifact)}
            if task["instance_id"] == "collection":
                result["tests"] = {"t": "error"}
                result["collection_error"] = "import failed"
            else:
                result["tests"] = {"t": "skipped"}
            return result

        result = ReproEvaluator(runner).evaluate(
            [{"instance_id": "collection"}, {"instance_id": "skipped"}],
            {"collection": self._artifact(), "skipped": self._artifact()},
        )
        records = {row["instance_id"]: row for row in result["records"]}
        self.assertFalse(records["collection"]["valid_reproduction"])
        self.assertFalse(records["skipped"]["valid_reproduction"])
        self.assertEqual(records["collection"]["failure_category"], "test_error")
        self.assertEqual(records["skipped"]["failure_category"], "all_tests_skipped")

    def test_timeout_and_invalid_are_explicit_and_deterministic(self):
        def runner(task, version, artifact):
            if task["instance_id"] == "timeout":
                return {"status": "timeout", "artifact_sha256": artifact_sha256(artifact)}
            return {"status": "completed", "tests": {"t": "passed"}, "artifact_sha256": artifact_sha256(artifact)}

        result = ReproEvaluator(runner).evaluate(
            [{"instance_id": "timeout"}, {"instance_id": "invalid"}],
            {"timeout": self._artifact(), "invalid": None},
        )
        self.assertEqual([r["status"] for r in result["records"]], ["model_invalid", "timeout"])

    def test_dry_run_does_not_invoke_runner(self):
        called = []
        result = ReproEvaluator(lambda *args: called.append(args)).evaluate(
            [{"instance_id": "x"}], {"x": {"tests": ["t"]}}, dry_run=True
        )
        self.assertEqual(result["records"][0]["status"], "dry_run")
        self.assertFalse(called)
        self.assertIsNone(result["aggregate"]["valid_reproduction_rate"])


if __name__ == "__main__":
    unittest.main()
