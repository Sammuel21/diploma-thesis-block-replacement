"""Read-only historical schema and adapter contracts; never loads checkpoint weights."""

import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
REFERENCES = (
    ("data/results/workflows/model/swiglu-3/run-001.json",
     "105e6e0588fb0281cfa382d5c47daa976c1ab00dbf504a132676cf037364d3da", "swiglu-3"),
    ("data/results/workflows/model/swiglu-4/run-001.json",
     "b3b00b92a2b291c1bd36e7c9d1ef72bdeaa9b4914492b77f5bc7f344c04695ac", "swiglu-4"),
    ("data/results/workflows/model/swiglu-5/search/run-001.json",
     "28f7e5624bb490f1dc1bd87d63695917ddd684dc95464642da9b04689680715b", "swiglu-5-search"),
    ("data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.2.json",
     "940677d6853e9df98b0136e84ea939247bdd43a8591aa27faf27056e2310b180", "swiglu-5-confirmation"),
    ("data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.5.json",
     "47f1bc98b32d4bd8b1935daaf7880c7cb94ad018f74bfa02c575b6bcc33614e4", "swiglu-5-confirmation"),
)


class HistoricalArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [path for path, digest, workflow in REFERENCES if not (ROOT / path).is_file()]
        if missing:
            raise unittest.SkipTest("Ignored historical JSON artifacts unavailable: " + ", ".join(missing))
        cls.records = [json.loads((ROOT / path).read_text(encoding="utf-8"))
                       for path, digest, workflow in REFERENCES]

    def test_reference_bytes_and_schemas(self):
        for (path, digest, workflow), record in zip(REFERENCES, self.records):
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
                self.assertEqual(record["schema_version"], 1)
                self.assertEqual(record["workflow"], workflow)
                self.assertEqual(record["status"], "completed")

    def test_source_adapters_and_retained_state_interpretation(self):
        from workflows.runs.model.swiglu.shared import source_allocation, source_milestone, source_operator_rows
        from workflows.runs.model.swiglu.swiglu5.confirmation import confirmation_candidate

        source, s4, search, *confirmations = self.records
        self.assertEqual(s4["results"]["source_state"]["sparsity_key"], "0.5")
        self.assertEqual(source_milestone(source, "0.5", 10_000_000),
                         s4["results"]["source_state"]["swiglu_3_10m"])
        for target, confirmation in zip((0.2, 0.5), confirmations):
            key = str(target)
            selected = source_allocation(source, key)
            expected = [row for row in source["results"]["sparsity"]["allocation"]
                        if row["sparsity_key"] == key]
            self.assertEqual(selected, expected)
            self.assertIsNot(selected[0], expected[0])
            operators, allocation = source_operator_rows(source, 393216, key)
            self.assertEqual(list(allocation), [int(row["layer"]) for row in expected])
            self.assertEqual(set(operators), set(allocation))
            selection = search["results"]["selection"][key]
            self.assertEqual(selection["winner_candidate_id"], "S5-C2")
            self.assertEqual(selection["winner_endpoint"]["actual_tokens"], 5_001_216)
            self.assertTrue(selection["winner_endpoint"]["retained"])
            context = SimpleNamespace(search=search, settings={
                "target": target, "selected_candidate_id": selection["winner_candidate_id"],
            })
            candidate = confirmation_candidate(context)
            self.assertIs(candidate, search["results"]["candidates"][key]["S5-C2"])
            self.assertTrue(all({"layer", "replacement_width", "retains_dense_module", "has_output_bias"}
                                <= row.keys() for row in candidate["allocation"]))
            result = confirmation["results"]
            self.assertEqual((result["target"], result["selected_candidate_id"]), (target, "S5-C2"))
            trajectory = result["trajectory"]
            self.assertEqual(trajectory["checkpoint_policy"], "final_and_best_weights_only_no_resume")
            self.assertEqual(trajectory["final_checkpoint"]["tokens_seen"], 100_000_000)
            self.assertEqual(trajectory["best_checkpoint"]["tokens_seen"], trajectory["best_checkpoint_tokens"])
            for state in (trajectory["final_checkpoint"], trajectory["best_checkpoint"]):
                self.assertEqual(state["contents"], "replacement_weights_only")
                self.assertEqual(len(state["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
