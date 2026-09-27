"""The session digest: Laya-g1's only input. Fixed keys and order, bounded length, no model names."""
import json
import unittest
from pathlib import Path

from router.digest import BUDGET_WORDS, build, classify_tool, error_evidence, is_test_pass, render

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "digest"


def call(name, args="{}"):
    return {"type": "function", "function": {"name": name, "arguments": args}}


class ToolVocabularyParity(unittest.TestCase):
    """Cases from dss-gateway's router.rs (builtin_semantic) and its doc comment."""

    def test_the_built_in_vocabulary(self):
        cases = {"KB_search": "observe", "send_payment_request": "mutate", "mcp.erp2.list_documents": "observe",
                 "memory_note_write": "mutate", "frobnicate": "unknown", "read_file": "observe", "Read": "observe",
                 "run_tests": "test", "testing_harness": "test", "bash": "shell", "todo_write": "plan", "cat": "observe"}
        self.assertEqual({k: classify_tool(k) for k in cases}, cases)

    def test_error_evidence_and_test_pass(self):
        self.assertEqual(error_evidence("Traceback (most recent call last):\n  File"), "critical")
        self.assertEqual(error_evidence("grep: foo: No such file or directory"), "some")
        self.assertEqual(error_evidence("<returncode>0</returncode>"), "none")
        self.assertTrue(is_test_pass("===== 12 passed in 0.3s ====="))
        self.assertFalse(is_test_pass("3 passed, 1 failed"))


class Digest(unittest.TestCase):
    def agentic(self):
        return [
            {"role": "system", "content": "You are a helpful assistant that can interact with a computer shell to solve programming tasks."},
            {"role": "user", "content": "Fix model_to_dict() so it returns an empty dict for an empty list of fields."},
            {"role": "assistant", "content": "Let me find the function.", "tool_calls": [call("bash", '{"command": "grep -r model_to_dict ."}')]},
            {"role": "tool", "tool_call_id": "1", "content": "./django/forms/models.py:def model_to_dict(instance, fields=None):"},
            {"role": "assistant", "content": "Now I'll run the tests.", "tool_calls": [call("run_tests")]},
            {"role": "tool", "tool_call_id": "2", "content": "Traceback (most recent call last):\n  File x"},
        ]

    def test_keys_are_fixed_and_ordered(self):
        d = build(self.agentic())
        self.assertEqual(list(d), ["harness", "goal", "request", "prior_user", "tools_offered", "tool_rounds", "progress", "context", "turn"])
        self.assertEqual(d["tool_rounds"]["n"], 2)
        self.assertEqual(d["tool_rounds"]["last"], ["test"])
        self.assertTrue(d["tool_rounds"]["critical"])
        self.assertEqual(d["progress"], "Now I'll run the tests.")
        self.assertEqual(d["goal"], "Fix model_to_dict() so it returns an empty dict for an empty list of fields.")
        self.assertEqual(d["request"], "", "the only user turn is the goal: not repeated")

    def test_a_follow_up_is_read_against_its_goal(self):
        msgs = [{"role": "user", "content": "Migrate the billing service from MySQL to Postgres without downtime."},
                {"role": "assistant", "content": "Here is a four-step plan..."},
                {"role": "user", "content": "yes, go ahead"}]
        d = build(msgs)
        self.assertEqual(d["goal"], "Migrate the billing service from MySQL to Postgres without downtime.")
        self.assertEqual(d["request"], "yes, go ahead")
        self.assertEqual(d["turn"], 2)

    def test_offered_tools_are_counted_by_category(self):
        tools = [{"type": "function", "function": {"name": n}} for n in ("read_file", "write_file", "bash", "grep")]
        d = build([{"role": "user", "content": "hi"}], tools=tools)
        self.assertEqual(d["tools_offered"], {"n": 4, "categories": {"mutate": 1, "observe": 2, "shell": 1}})

    def test_catalogue_names_never_reach_the_digest(self):
        msgs = [{"role": "system", "content": "Available models: claude-opus-5-5, gpt-6-sol, qwen3.8-27b. Pick wisely."},
                {"role": "user", "content": "Ask gpt-6-sol to summarise this."}]
        text = render(build(msgs, redact=["claude-opus-5-5", "gpt-6-sol", "qwen3.8-27b"]))
        for name in ("claude-opus-5-5", "gpt-6-sol", "qwen3.8-27b"):
            self.assertNotIn(name, text)
        self.assertIn("<model>", text)

    def test_length_is_bounded_whatever_the_session(self):
        huge = " ".join(f"w{i}" for i in range(20000))
        msgs = [{"role": "system", "content": huge}, {"role": "user", "content": huge}]
        for i in range(40):
            msgs += [{"role": "assistant", "content": huge, "tool_calls": [call(f"tool_{i}")]}, {"role": "tool", "content": huge}]
        msgs.append({"role": "user", "content": huge})
        self.assertLessEqual(len(render(build(msgs)).split()), BUDGET_WORDS)

    def test_content_parts_are_read_as_text(self):
        d = build([{"role": "user", "content": [{"type": "text", "text": "Describe"}, {"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "this chart"}]}])
        self.assertEqual(d["goal"], "Describe this chart")

    def test_golden_fixtures_are_byte_identical(self):
        """The contract the Rust builder (dss-gateway, S8) must reproduce."""
        paths = sorted(FIXTURES.glob("*.json"))
        self.assertGreaterEqual(len(paths), 4)
        for path in paths:
            with self.subTest(path.name):
                case = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(render(build(case["messages"], tools=case.get("tools"), redact=case.get("redact", []))), case["digest"])


if __name__ == "__main__":
    unittest.main()
