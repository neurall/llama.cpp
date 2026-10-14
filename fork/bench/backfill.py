#!/usr/bin/env python3
"""Rebuild tools/bench/stats.csv from stats.ini + run-history.csv and fill the branch / patch columns of runs made before run.py recorded them.

usage: backfill.py            (idempotent: rows already in stats.csv are skipped)
Rules for old rows (no BUILD_INFO): dev / pred builds of 2026-10-02 were built from the release branch with an uncommitted tree (patch "uncommitted"), except campaigns
listed in EXP, which ran on their experiment branch at the given commit.
"""
import configparser, csv, io, json, os, sys
sys.argv = sys.argv[:1]
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("run", os.path.join(HERE, "..", "run.py")); run = importlib.util.module_from_spec(sp); sp.loader.exec_module(run)
EXP = {"lead-ab-pin": ("f12b895ab", "predictor-auto-layer-offset")}   # campaign -> (commit, branch) of a clean experiment branch

rows = list(csv.reader(open(os.path.join(HERE, "run-history.csv"))))
h = rows[0]
for c in run.RUN_COLS:
    if c not in h:
        h.append(c)
ix = {c: h.index(c) for c in h}
for r in rows[1:]:
    r += [""] * (len(h) - len(r))
    if r[ix["build"]] in ("dev", "pred") and r[ix["ts"]] >= "2026-10-02" and not r[ix["branch"]]:
        camp = r[ix["campaign"]]
        if camp in EXP:
            r[ix["commit"]], r[ix["branch"]], r[ix["patch"]] = EXP[camp][0], EXP[camp][1], "0"
        else:
            r[ix["branch"]], r[ix["patch"]] = "release", "uncommitted"
    while len(r) > len(run.RUN_COLS) and r[-1] == "":
        r.pop()
# write: header with all columns, rows as they are (shorter rows stay shorter)
out = io.StringIO(); w = csv.writer(out, lineterminator="\n"); w.writerow(run.RUN_COLS); [w.writerow(r) for r in rows[1:]]
open(os.path.join(HERE, "run-history.csv"), "w").write(out.getvalue())

byid = {}
for r in rows[1:]:
    d = dict(zip(h, r + [""] * (len(h) - len(r))))
    byid[(d["ts"], d["build"])] = d
ini = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";",)); ini.read(os.path.join(HERE, "stats.ini"))
have = set()
p = os.path.join(HERE, "stats.csv")
if os.path.exists(p):
    have = {(r["ts"], r["build"]) for r in csv.DictReader(open(p))}
names = ["tok_s", "hit_pct", "uploads", "evictions", "up_mib", "ddr_gbs", "pcie_gbs", None, "pred_up", "pred_pub", "pred_used", "pred_late", "layer_ms_avg", "layer_ms_min", "up_ms_l0", "up_ms_l1"]
n = 0
for sec in ini.sections():
    ts, _, build = sec.rpartition(" ")
    d = byid.get((ts, build))
    if d is None or (ts, build) in have:
        continue
    s = ini[sec]
    every = int(s["every"].split()[0])
    cols = {k: [float(x) for x in s[k].split(",")] for k in s if k not in ("model", "test", "hw", "every")}
    nw = len(cols["tok_s"])
    wnd = []
    for i in range(nw):
        v = [i * every, 0.0, cols["tok_s"][i], cols["hit_pct"][i], cols["uploads"][i], cols["evictions"][i], cols["up_mib"][i], cols["ddr_gbs"][i], cols["pcie_gbs"][i], 0.0]
        if "pred_up" in cols:
            v += [cols[k][i] for k in ("pred_up", "pred_pub", "pred_used", "pred_late")]
        if "layer_ms_avg" in cols:
            v += [cols[k][i] for k in ("layer_ms_avg", "layer_ms_min", "up_ms_l0", "up_ms_l1")]
        wnd.append(v)
    env = {k: v for k, v in json.loads(d["env"] or "{}").items()}
    run.csv_append("stats.csv", run.STATS_CSV_COLS, dict(ts=ts, build=build, commit=d["commit"], branch=d["branch"], patch=d["patch"], hw=d["hw"], model=d["model"], test=d["test"], args=d["args"],
                   env=json.dumps(env, sort_keys=True, separators=(",", ":")) if env else "", tps=d["tps"], hit=d["hit"], **run.stats_csv_row(wnd)))
    n += 1
print(n, "rows added to stats.csv")
