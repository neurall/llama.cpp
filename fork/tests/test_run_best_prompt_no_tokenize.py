"""run.py best_prompt_chat: an engine without /tokenize and /apply-template (Strata answers HTTP 404) gets the stored best prompt as it is: the file
fork/prompts/<name>.json was cut to the exact token count by a llama.cpp server with the same tokenizer. Without that fallback every Strata cell of
gainers2 failed with "HTTP Error 404: Not Found" (2026-11-18, rows 30892-30895).
run: python3 fork/tests/test_run_best_prompt_no_tokenize.py   (no GPU; imports run.py)"""
import importlib.util, json, os, tempfile, unittest, urllib.error

RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run.py")
_spec = importlib.util.spec_from_file_location("run_mod", RUN_PY)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


class NoTokenize(unittest.TestCase):
    def test_stored_prompt_is_used_when_the_server_has_no_tokenize(self):
        d = tempfile.mkdtemp(); run.PROMPT_DIR = d
        json.dump({"messages": [{"role": "user", "content": "stored text"}], "n": 100}, open(os.path.join(d, "p-x.json"), "w"))
        def post(path, body): raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        self.assertEqual(run.best_prompt_chat(post, "other", 100, "p-x"), "stored text")


if __name__ == "__main__":
    unittest.main()
