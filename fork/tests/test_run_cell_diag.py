"""run.py: a cell's diagnostics (DIAG_ENV: pool write ledger, live pool view, scratch check; each costs speed) come from its command line or the
caller's environment, never from a build's name, and every one that is on is said. A fix_ build name turned the ledger on silently and cost
about 25% decode in a fix retest (2026-11-14).
run: python3 fork/tests/test_run_cell_diag.py   (no GPU; imports run.py, about 10 s)"""
import argparse
import importlib.util
import os
import re
import unittest

RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run.py")
_spec = importlib.util.spec_from_file_location("run_mod", RUN_PY)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def cell(builds="x", ledger=False, env=()):
    return argparse.Namespace(builds=builds, ledger=ledger, env=list(env))


class CellDiag(unittest.TestCase):
    def test_build_name_never_turns_a_diagnostic_on(self):
        for b in ("fix_staging_one_reader", "ledger_check", "scratch_x", "pool_live", "unread_fixes", "fix_a,fix_b"):
            env, on = run.cell_diag({}, cell(builds=b))
            self.assertEqual(on, [], b)
            self.assertTrue(all(env.get(k) in (None, "", "0") for k in run.DIAG_ENV), b)

    def test_flag_turns_the_ledger_on_and_says_so(self):
        env, on = run.cell_diag({}, cell(ledger=True))
        self.assertEqual(env["PERF_LEDGER"], "1")
        self.assertTrue(any(s.startswith("PERF_LEDGER=1") for s in on), on)

    def test_every_diagnostic_on_is_said(self):
        for k in run.DIAG_ENV:   # by --env and by the caller's environment
            self.assertTrue(any(s.startswith(k + "=1") for s in run.cell_diag({}, cell(env=[k + "=1"]))[1]), k)
            self.assertTrue(any(s.startswith(k + "=1") for s in run.cell_diag({k: "1"}, cell())[1]), k)

    def test_env_off_wins_over_nothing(self):
        env, on = run.cell_diag({"PERF_LEDGER": "1"}, cell(env=["PERF_LEDGER=0"]))
        self.assertEqual(on, [])

    def test_no_diagnostic_is_set_outside_cell_diag(self):
        """Every write of a DIAG_ENV key in run.py is inside cell_diag: a rule elsewhere (a build name, a test name, a machine) is a hidden switch."""
        with open(RUN_PY) as f: src = f.read().split("\n")
        keys = "|".join(map(re.escape, run.DIAG_ENV))
        write = re.compile(r'(\[\s*"(%s)"\s*\]\s*=[^=])|(setdefault\(\s*"(%s)")|(\b(%s)\s*=\s*["\']1)' % (keys, keys, keys))
        inside, bad = False, []
        for n, line in enumerate(src, 1):
            if line.startswith("def "):
                inside = line.startswith("def cell_diag(")
            if write.search(line) and not inside and not line.lstrip().startswith("#"):
                bad.append(f"run.py:{n}: {line.strip()[:120]}")
        self.assertEqual(bad, [], "\n".join(bad))

    def test_no_build_name_rule(self):
        with open(RUN_PY) as f: self.assertNotIn('startswith("fix_")', f.read())


if __name__ == "__main__":
    unittest.main()
