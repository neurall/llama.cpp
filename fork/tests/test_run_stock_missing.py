"""run.py: `run LIST -nrt` runs the stock cell of a row whose stock comparison is missing on this machine. The check read hw='pc1' alone, but pc1 is
two machines (the Ryzen 3700X desktop until 2026-11-11, the Threadripper PRO 3975WX from 2026-11-12, HW_SINCE): on 2026-11-18 `run gainers2 -nrt` planned
no stock cell for GLM 5.3 3.5-bit although the workstation has none, because the desktop's stock rows of 10-23 counted. The check must read the rows of
the current hardware only (hw_filter: the date scope and one request at a time).
run: python3 fork/tests/test_run_stock_missing.py   (no GPU; imports run.py, about 10 s)"""
import importlib.util
import os
import sqlite3
import unittest

RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run.py")
_spec = importlib.util.spec_from_file_location("run_mod", RUN_PY)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


class StockMissing(unittest.TestCase):
    def db(self, ts, args="-np 1 -c 16384"):
        c = sqlite3.connect(":memory:")
        c.execute("create table runs (model text, test text, build text, ok text, tps real, env text, args text, hw text, n_gen real, ts text)")
        c.execute("insert into runs values ('GLM.gguf', 'pf12k', 'stock-b11829-official', '1', 10.7, '', ?, 'pc1', 1024, ?)", (args, ts))
        return c

    def found(self, c, gen=1024):
        run.HWF[0] = run.hw_filter("pc1")
        return c.execute(run.stock_cell_sql(gen), ("GLM.gguf", "pf12k")).fetchone() is not None

    def test_old_machine_rows_do_not_count(self):
        self.assertFalse(self.found(self.db("2026-10-23 10:00:00")))

    def test_rows_of_this_machine_count(self):
        self.assertTrue(self.found(self.db("2026-11-16 10:00:00")))

    def test_parallel_rows_do_not_count(self):
        self.assertFalse(self.found(self.db("2026-11-16 10:00:00", "-np 4 -c 32768")))

    def test_answer_length_still_matters(self):
        self.assertFalse(self.found(self.db("2026-11-16 10:00:00"), gen=64))


if __name__ == "__main__":
    unittest.main()
