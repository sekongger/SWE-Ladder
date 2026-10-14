import tempfile
import unittest

from stage_capability_eval.preflight import (
    PreflightRecord,
    apply_gold_patch,
    normalize_command_result,
    parse_test_names,
)


class PreflightTests(unittest.TestCase):
    def test_pass_and_fail_are_distinct(self):
        passed = normalize_command_result(returncode=0, stdout="ok", stderr="")
        failed = normalize_command_result(returncode=1, stdout="", stderr="assertion")
        self.assertEqual(passed.status, "passed")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.failure_category, "test_failure")

    def test_timeout_and_infra_are_not_test_failures(self):
        timeout = normalize_command_result(returncode=None, stdout="", stderr="", timed_out=True)
        infra = normalize_command_result(returncode=None, stdout="", stderr="docker unavailable", infra_error=True)
        self.assertEqual(timeout.status, "timeout")
        self.assertEqual(timeout.failure_category, "timeout")
        self.assertEqual(infra.status, "infra_error")
        self.assertEqual(infra.failure_category, "infrastructure")

    def test_empty_output_and_invalid_returncode(self):
        empty = normalize_command_result(returncode=0, stdout="", stderr="")
        invalid = normalize_command_result(returncode=None, stdout="", stderr="")
        self.assertEqual(empty.status, "passed")
        self.assertEqual(invalid.status, "invalid")

    def test_parse_test_names(self):
        self.assertEqual(parse_test_names('["a::test_x", "b::test_y"]'), ["a::test_x", "b::test_y"])
        self.assertEqual(parse_test_names(""), [])

    def test_record_is_json_serializable(self):
        record = PreflightRecord(instance_id="x", phase="buggy", status="passed")
        self.assertEqual(record.to_dict()["instance_id"], "x")

    def test_patch_apply_reports_invalid_patch_without_mutating(self):
        with tempfile.TemporaryDirectory() as directory:
            result = apply_gold_patch(directory, "this is not a unified diff\n")
            self.assertEqual(result.status, "invalid")
            self.assertEqual(result.failure_category, "patch_apply")


if __name__ == "__main__":
    unittest.main()
