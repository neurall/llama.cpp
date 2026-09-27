import json, sys
t = json.load(sys.stdin)["timings"]
print(f"{t['predicted_n']} tok {t['predicted_per_second']:.1f} t/s")
