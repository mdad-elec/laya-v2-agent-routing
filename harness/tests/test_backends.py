import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from harness.backends import GatewayBackend
from harness.runner import SeatLimit

CATALOGUE = {"models": [{"model": "claude-code/claude-opus-5-5", "efforts": ["low", "high"]},
                        {"model": "pair/qwen2.5:3b", "efforts": ["none"]},
                        {"model": "codex/gpt-5.6-sol", "efforts": ["none", "low", "high"]}]}


class Fake(BaseHTTPRequestHandler):
    seen, mode = [], "ok"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        Fake.seen.append((dict(self.headers), body))
        if Fake.mode == "limit":
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error":"usage limit reached"}')
            return
        if Fake.mode == "broken":
            self.send_response(502)
            self.end_headers()
            self.wfile.write(b'{"error":"seat crashed"}')
            return
        out = {"choices": [{"message": {"content": "hi from " + body["model"]}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())

    def log_message(self, *a):
        pass


class Gateway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        os.environ["LAYA_TEST_GATEWAY_TOKEN"] = "Bearer secret-token"
        cls.b = GatewayBackend(f"http://127.0.0.1:{cls.server.server_port}", token_env="LAYA_TEST_GATEWAY_TOKEN", catalogue=CATALOGUE)

    def setUp(self):
        Fake.seen.clear()
        Fake.mode = "ok"

    def test_a_cell_becomes_a_named_model_call_at_its_effort(self):
        text, usage = self.b.ask("claude-code/claude-opus-5-5@high", [{"role": "user", "content": "x"}])
        headers, body = Fake.seen[-1]
        self.assertEqual((body["model"], body["reasoning_effort"]), ("claude-opus-5-5", "high"))
        self.assertEqual(headers["Authorization"], "Bearer secret-token")
        self.assertEqual(text, "hi from claude-opus-5-5")
        self.assertEqual((usage["input_tokens"], usage["output_tokens"]), (7, 3))
        self.assertIn("latency_ms", usage)

    def test_none_is_no_field_for_a_model_without_efforts_and_a_real_id_for_one_that_declares_it(self):
        self.b.ask("pair/qwen2.5:3b@none", [{"role": "user", "content": "x"}])
        self.assertNotIn("reasoning_effort", Fake.seen[-1][1])
        self.b.ask("codex/gpt-5.6-sol@none", [{"role": "user", "content": "x"}])
        self.assertEqual(Fake.seen[-1][1]["reasoning_effort"], "none")

    def test_a_seat_limit_raises_seatlimit_and_any_other_failure_raises_with_its_text(self):
        Fake.mode = "limit"
        with self.assertRaises(SeatLimit):
            self.b.ask("claude-code/claude-opus-5-5@low", [{"role": "user", "content": "x"}])
        Fake.mode = "broken"
        with self.assertRaisesRegex(RuntimeError, "502.*seat crashed"):
            self.b.ask("claude-code/claude-opus-5-5@low", [{"role": "user", "content": "x"}])

    def test_an_effort_the_model_does_not_declare_is_refused_before_any_call(self):
        with self.assertRaisesRegex(ValueError, "ultra"):
            self.b.ask("claude-code/claude-opus-5-5@ultra", [{"role": "user", "content": "x"}])
        self.assertEqual(Fake.seen, [])

    def test_the_token_never_appears_in_an_error(self):
        Fake.mode = "broken"
        try:
            self.b.ask("claude-code/claude-opus-5-5@low", [{"role": "user", "content": "x"}])
        except RuntimeError as e:
            self.assertNotIn("secret-token", str(e))


if __name__ == "__main__":
    unittest.main()
