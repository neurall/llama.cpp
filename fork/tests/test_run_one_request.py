"""run.py: best and reference lookups read rows of one request at a time only. A `--par N` row (N requests at once, server -np N) stores the aggregate
of all N answers as its t/s: the 2026-10-16 par probe put 101-131 t/s "decode" for Qwen3.8 Next IQ3_S under the plain test names tetrisc and game4c
(single-request rows: 55-62), and runlog's consistent best of that build and cell came out as the -np 4 row (user 2026-11-18: "i always demanded single
req /slot test").
run: python3 fork/tests/test_run_one_request.py   (no GPU; imports run.py, about 10 s)"""
import importlib.util
import os
import sqlite3
import unittest

RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run.py")
_spec = importlib.util.spec_from_file_location("run_mod", RUN_PY)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


class OneRequest(unittest.TestCase):
    def rows(self, where):
        c = sqlite3.connect(":memory:")
        c.execute("create table runs (args text, hw text, ts text)")
        for a in ("-np 1 -c 1024", "", None, "-np 4 -c 32768", "-np 10 -c 32768", "-np 12", "-np 1 --moe staging=0"):
            c.execute("insert into runs values (?, 'pc1', '2026-11-18')", (a,))
        return [r[0] for r in c.execute("select args from runs where 1=1" + where)]

    def test_parallel_rows_are_left_out(self):
        self.assertEqual(self.rows(run.ONE_REQ), ["-np 1 -c 1024", "", None, "-np 1 --moe staging=0"])

    def test_every_lookup_of_a_list_has_it(self):   # HWF[0] = hw_filter(...) is the filter of every best / reference / stock query of a list
        for hw in ("pc1", "pc2"):
            self.assertIn(run.ONE_REQ, run.hw_filter(hw), hw)
            self.assertNotIn("-np 4 -c 32768", self.rows(run.hw_filter(hw)), hw)


if __name__ == "__main__":
    unittest.main()
