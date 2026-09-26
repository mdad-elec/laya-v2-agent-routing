import json
import unittest

from suites.bfcl.suite import BFCL, decode_calls, system_prompt


class Decode(unittest.TestCase):
    def test_a_python_style_call_list_decodes_to_call_strings(self):
        self.assertEqual(decode_calls("[cd(folder='document'), mkdir(dir_name='temp')]"), ["cd(folder='document')", "mkdir(dir_name='temp')"])
        self.assertEqual(decode_calls("```python\n[ls()]\n```"), ["ls()"])
        self.assertEqual(decode_calls("[sort('final_report.pdf')]"), ["sort('final_report.pdf')"], "positional, as BFCL's own ground truth")
        self.assertEqual(decode_calls("[get_stock_info(symbol='AAPL', details={'x': [1, 2]})]"), ["get_stock_info(symbol='AAPL', details={'x': [1, 2]})"])

    def test_anything_that_is_not_a_list_of_plain_calls_decodes_to_nothing(self):
        for text in ("I have finished the task.", "", "[__import__('os').system('rm -rf /')]", "[a.b()]", "[f(*x)]", "[f(x)]", "[1, 2]", "print(1)"):
            self.assertEqual(decode_calls(text), [], text)


class Episodes(unittest.TestCase):
    def test_the_system_prompt_is_bfcls_default_with_the_function_docs(self):
        text = system_prompt([{"name": "ls", "description": "list", "parameters": {}}])
        self.assertTrue(text.startswith("You are an expert in composing functions."))
        self.assertIn("[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]", text)
        self.assertIn('"name": "ls"', text)

    def test_replaying_the_ground_truth_passes_and_a_wrong_call_fails(self):
        s = BFCL()
        items = s.load()
        item = items[0]
        turns = item.gold["ground_truth"]

        def replay(truth):
            state = {"turn": 0, "sent": False}

            def ask(messages):
                user_turns = sum(1 for m in messages if m["role"] == "user" and not m["content"].startswith("Execution results"))
                t = user_turns - 1
                if state["turn"] != t:
                    state.update(turn=t, sent=False)
                if not state["sent"]:
                    state["sent"] = True
                    return "[" + ", ".join(truth[t]) + "]"
                return "Done."
            return ask

        score, transcript = s.run(item, replay(turns))
        self.assertEqual(score, 1.0, json.dumps(transcript)[:600])
        wrong = [list(t) for t in turns]
        wrong[0] = ["ls()"]
        self.assertEqual(s.run(item, replay(wrong))[0], 0.0)


class Golden(unittest.TestCase):
    """The loop a model runs, fed BFCL's own ground truth, is valid on all 200 episodes: the
    decoder, the executor wiring, the per-run state isolation and the checker agree with BFCL."""

    def test_every_ground_truth_episode_replays_valid(self):
        import os

        if os.environ.get("LAYA_OFFLINE") == "1":
            self.skipTest("offline")
        s = BFCL()
        items = s.load()
        self.assertEqual(len(items), 200)

        def replay(truth):
            st = {"turn": -1, "sent": False}

            def ask(messages):
                t = sum(1 for m in messages if m["role"] == "user" and not m["content"].startswith("Execution results")) - 1
                if st["turn"] != t:
                    st.update(turn=t, sent=False)
                if not st["sent"] and truth[t]:
                    st["sent"] = True
                    return "[" + ", ".join(truth[t]) + "]"
                return "Done."
            return ask

        failures = [i.native_id for i in items if s.run(i, replay(i.gold["ground_truth"]))[0] != 1.0]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
