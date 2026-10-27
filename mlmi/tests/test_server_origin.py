"""Only Sheep's own page may drive the backend: a web page elsewhere can POST to
127.0.0.1, and a rebinding domain can resolve there, so both are refused."""
import http.client
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import server  # noqa: E402


class OriginGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state = SimpleNamespace(orch=SimpleNamespace(
            retry_orientation_failure=lambda: {"ok": True, "task_id": "x"}))
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(state))
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, **headers):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        host = headers.pop("Host", None)
        c.putrequest("POST", "/api/orient/retry", skip_host=host is not None)
        if host is not None:
            c.putheader("Host", host)
        for k, v in dict({"Content-Length": "2"}, **headers).items():
            c.putheader(k.replace("_", "-"), v)
        c.endheaders(b"{}")
        status = c.getresponse().status
        c.close()
        return status

    def test_own_page_and_local_tools_pass(self):
        self.assertEqual(self.post(), 202)
        self.assertEqual(self.post(Origin=f"http://127.0.0.1:{self.port}"), 202)
        self.assertEqual(self.post(Origin=f"http://localhost:{self.port}", Host=f"localhost:{self.port}"), 202)

    def test_other_sites_are_refused(self):
        self.assertEqual(self.post(Origin="https://evil.example"), 403)
        self.assertEqual(self.post(Origin="null"), 403)
        self.assertEqual(self.post(Origin=f"http://127.0.0.1:{self.port + 1}"), 403)
        self.assertEqual(self.post(Sec_Fetch_Site="cross-site"), 403)

    def test_rebinding_host_is_refused(self):
        self.assertEqual(self.post(Host=f"evil.example:{self.port}"), 403)


if __name__ == "__main__":
    unittest.main()
