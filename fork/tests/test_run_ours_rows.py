"""run.py: rows of another engine (Strata, builds named strata-*, user 2026-11-18: "add strata runs next to stock for 3 way comparison") are neither
stock nor ours. Every lookup of our own best, reference or older runs reads `build not like 'stock%'`; a Strata row would pass that and become
"our best" for its cell (its Qwen3.8 Next decode is about 1.5x ours, so every later run of ours would show a false regression).
run: python3 fork/tests/test_run_ours_rows.py   (no GPU; imports run.py, about 10 s)"""
import importlib.util
import os
import re
import sqlite3
import unittest

RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run.py")
_spec = importlib.util.spec_from_file_location("run_mod", RUN_PY)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


class OursRows(unittest.TestCase):
    def test_other_engines_are_not_ours(self):
        c = sqlite3.connect(":memory:")
        c.execute("create table runs (build text)")
        for b in ("rel_pool_copy_own_card", "stock-b12499-official", "strata-0.2.15", "strata-0.2.15-full"):
            c.execute("insert into runs values (?)", (b,))
        self.assertEqual([r[0] for r in c.execute("select build from runs where 1=1" + run.OURS_SQL)], ["rel_pool_copy_own_card"])

    def test_every_ours_lookup_leaves_them_out(self):
        src = open(RUN_PY).read()
        self.assertEqual(re.findall(r".{40}not like 'stock%'(?! and build not like 'strata%')", src), [])


if __name__ == "__main__":
    unittest.main()
