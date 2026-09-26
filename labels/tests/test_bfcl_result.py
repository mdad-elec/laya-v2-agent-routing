import json
import unittest

from labels.bfcl_result import outcomes_from_score_file


class BFCLResult(unittest.TestCase):
    def test_a_score_file_becomes_one_pass_or_fail_per_entry_and_its_own_count_is_checked(self):
        lines = [json.dumps({"accuracy": 0.99, "correct_count": 198, "total_count": 200}),
                 json.dumps({"id": "multi_turn_base_1", "valid": False, "error": {"error_type": "multi_turn:force_terminated"}}),
                 json.dumps({"id": "multi_turn_base_7", "valid": False, "error": {"error_type": "state_mismatch"}})]
        rows = outcomes_from_score_file("\n".join(lines), model="m-FC", n_entries=200)
        self.assertEqual(len(rows), 200)
        self.assertEqual(sum(r["pass"] for r in rows), 198)
        failed = {r["native_id"]: r["reason"] for r in rows if not r["pass"]}
        self.assertEqual(failed, {"multi_turn_base_1": "multi_turn:force_terminated", "multi_turn_base_7": "state_mismatch"})

    def test_a_file_whose_failures_disagree_with_its_own_summary_is_refused(self):
        lines = [json.dumps({"accuracy": 1.0, "correct_count": 200, "total_count": 200}),
                 json.dumps({"id": "multi_turn_base_1", "valid": False, "error": {}})]
        with self.assertRaisesRegex(ValueError, "199"):
            outcomes_from_score_file("\n".join(lines), model="m", n_entries=200)


if __name__ == "__main__":
    unittest.main()
