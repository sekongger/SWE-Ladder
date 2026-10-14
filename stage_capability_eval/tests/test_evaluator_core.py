import unittest
import tempfile
from pathlib import Path

from stage_capability_eval.evaluator import (
    CheckpointModelAdapter,
    EvaluationError,
    InMemoryModelAdapter,
    StageCapabilityEvaluator,
)
from stage_capability_eval.loc_scorer import normalize_candidates, score_location


class LocationScorerTests(unittest.TestCase):
    def setUp(self):
        self.oracle = {"files": [{"path": "pkg/a.py", "functions": ["C.f"]}, {"path": "pkg/b.py", "functions": ["g"]}]}

    def test_full_hit_and_duplicate_normalization(self):
        candidates = normalize_candidates([{"file": "pkg/a.py", "function": "C.f"}, "pkg/a.py", {"path": "pkg/b.py", "functions": ["g"]}])
        self.assertEqual(candidates, [{"file": "pkg/a.py", "functions": ["C.f"]}, {"file": "pkg/b.py", "functions": ["g"]}])
        result = score_location(candidates, self.oracle, ks=(1, 2))
        self.assertEqual(result["file_acc@1"], 0.0)
        self.assertEqual(result["file_recall@1"], 0.5)
        self.assertEqual(result["function_acc@2"], 1.0)

    def test_partial_and_no_hit(self):
        partial = score_location([{"file": "pkg/a.py", "function": "wrong"}], self.oracle, ks=(1,))
        self.assertEqual(partial["file_acc@1"], 0.0)
        self.assertEqual(partial["function_acc@1"], 0.0)
        no_hit = score_location(["other.py"], self.oracle, ks=(1,))
        self.assertEqual(no_hit["file_acc@1"], 0.0)
        self.assertEqual(no_hit["file_recall@1"], 0.0)

    def test_protocol_top_level_files_and_functions(self):
        result = score_location(
            {"files": ["pkg/a.py"], "functions": [{"path": "pkg/a.py", "qualified_name": "C.f"}]},
            self.oracle,
            ks=(1,),
        )
        self.assertEqual(result["file_acc@1"], 0.0)
        self.assertEqual(result["file_recall@1"], 0.5)
        self.assertEqual(result["function_recall@1"], 0.5)


class EvaluatorAdapterTests(unittest.TestCase):
    def test_checkpoint_and_memory_adapters_have_same_schema(self):
        tasks = [{"instance_id": "one", "oracle": {"files": [{"path": "a.py", "functions": ["f"]}]}}]
        model = lambda task: [{"file": "a.py", "function": "f"}]
        memory = InMemoryModelAdapter(model)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "fake.ckpt"
            checkpoint_path.write_bytes(b"checkpoint")
            checkpoint = CheckpointModelAdapter(checkpoint_path, loader=lambda path: model)
            evaluator = StageCapabilityEvaluator(ks=(1,))
            a = evaluator.evaluate(tasks, memory, model_id="fake", seed=7, subset="dev")
            b = evaluator.evaluate(tasks, checkpoint, model_id="fake", seed=7, subset="dev")
        self.assertEqual(a["records"][0]["metrics"], b["records"][0]["metrics"])
        self.assertEqual(a["mode"], "in_memory")
        self.assertEqual(b["mode"], "checkpoint")
        self.assertEqual(a["records"][0]["seed"], 7)
        self.assertEqual(a["records"][0]["subset"], "dev")
        phases = [x["phase"] for x in a["logs"]]
        self.assertEqual((phases[0], phases[-1]), ("start", "end"))
        self.assertIn("progress", phases)

    def test_explicit_boundary_errors(self):
        with self.assertRaises(EvaluationError) as missing:
            CheckpointModelAdapter("/tmp/missing.ckpt")
        self.assertEqual(missing.exception.category, "missing_checkpoint")
        with self.assertRaises(EvaluationError) as empty:
            StageCapabilityEvaluator().evaluate([], InMemoryModelAdapter(lambda _: []))
        self.assertEqual(empty.exception.category, "empty_task_set")

    def test_model_invalid_stays_in_metric_denominator(self):
        tasks = [{"instance_id": "one", "oracle": {"files": [{"path": "a.py", "functions": ["f"]}]}}]
        result = StageCapabilityEvaluator(ks=(1,)).evaluate(
            tasks, InMemoryModelAdapter(lambda _: (_ for _ in ()).throw(ValueError("bad output")))
        )
        self.assertEqual(result["records"][0]["status"], "model_invalid")
        self.assertEqual(result["aggregate"]["file_acc@1"], 0.0)


if __name__ == "__main__":
    unittest.main()
