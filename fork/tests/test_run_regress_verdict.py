"""run.py: the release regress verdict reads the rows the regress run kept. `run regress -nrt` skips a cell that a build of the same commit already ran
today (the build number decides), so the verdict must count those rows too: on 2026-11-18 the release build rel_pool_copy_own_card skipped all 8 cells
(dev build pool_copy_own_card ran them at the same commit that morning) and then failed with "no valid ppl row". A build with uncommitted changes is other
code and never counts.
run: python3 fork/tests/test_run_regress_verdict.py   (no GPU; imports run.py, about 10 s)"""
import importlib.util
import io
import os
import unittest
from contextlib import redirect_stdout

RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run.py")
_spec = importlib.util.spec_from_file_location("run_mod", RUN_PY)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)
run.regress_promote = lambda *a: None   # a NEW BEST verdict copies the build under BEST: not from a test


def row(build, commit, tps, patch="0", model="m.gguf"):
    return dict(test="ppl", ok="1", tps=str(tps), build=build, commit=commit, patch=patch, model=model, args="")


def verdict(rows, commit="0d74ec7333"):
    out = io.StringIO()
    with redirect_stdout(out):
        rc = run.regress_list_verdict(rows, "release-0d74ec7333", "best_old", "/x/rel_new", 0.97, commit)
    return rc, out.getvalue()


class RegressVerdict(unittest.TestCase):
    def test_rows_of_a_build_of_the_same_commit_count(self):
        rc, out = verdict([row("pool_copy_own_card", "0d74ec7333", 30.0), row("pool_copy_own_card", "0d74ec7333", 31.0), row("best_old", "aaaa", 29.0)])
        self.assertNotIn("no valid ppl row", out)
        self.assertIn("pool_copy_own_card", out)   # says whose rows it read
        self.assertEqual(rc, 0, out)

    def test_a_patched_build_of_the_same_commit_does_not_count(self):
        rc, out = verdict([row("pool_copy_wip", "0d74ec7333", 30.0, patch="1a2b3c4d"), row("best_old", "aaaa", 29.0)])
        self.assertIn("no valid ppl row", out)
        self.assertEqual(rc, 2)

    def test_own_rows_still_count(self):
        rc, out = verdict([row("rel_new", "0d74ec7333", 30.0), row("best_old", "aaaa", 29.0)])
        self.assertEqual(rc, 0, out)

    def test_slower_rows_of_the_same_commit_still_fail(self):
        rc, out = verdict([row("pool_copy_own_card", "0d74ec7333", 20.0), row("best_old", "aaaa", 29.0)])
        self.assertNotEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main()
