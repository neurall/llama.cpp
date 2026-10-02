import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
names = {f.name for f in fm.fontManager.ttflist}
serif = next((n for n in ("Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif") if n in names), "serif")
plt.rcParams.update({"font.family": serif, "font.size": 8.5, "axes.linewidth": 0.7, "axes.labelsize": 9, "legend.fontsize": 7, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7, "savefig.dpi": 300, "mathtext.fontset": "dejavuserif"})
C_FORK, C_STOCK, C_GAIN, C_A, C_B = "#D55E00", "#0072B2", "#222222", "#009E73", "#CC79A7"   # Okabe-Ito
fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)

# machine D, GLM 3.0-bit: decode speed (left axis); share of the model in VRAM = GPUs x 22%
share = [22, 44, 66, 88]
fork = [22.9, 30.4, 32.6, 31.5]
stock = [14.6, 16.9, 20.1, 24.7]             # 22% and 44% modelled, 66% and 88% measured
ax.axvspan(30, 55, color="#DDDDDD", alpha=0.55, lw=0)
ax.axhline(35.3, color=C_FORK, lw=0.6, ls=":"); ax.text(101, 35.4, "fork floor", color=C_FORK, fontsize=6.3, ha="right", va="bottom")
ax.axhline(29.4, color=C_STOCK, lw=0.6, ls=":"); ax.text(101, 29.5, "stock floor", color=C_STOCK, fontsize=6.3, ha="right", va="bottom")
l1, = ax.plot(share, fork, "-s", color=C_FORK, ms=4.5, lw=1.2, zorder=3)
l2, = ax.plot(share, stock, "-o", color=C_STOCK, ms=4.5, lw=1.2, zorder=3)
ax.plot(share[:2], stock[:2], "o", mfc="white", mec=C_STOCK, ms=4.5, zorder=4)     # modelled points open
ax.set_xlim(15, 103); ax.set_ylim(8, 38)
ax.set_xlabel("Share of the model in VRAM (%)"); ax.set_ylabel("Decode speed, machine D, GLM 3.0-bit (tokens/s)")
ax.text(42.5, 37.5, "peak of the gain", ha="center", va="top", fontsize=6.5, color="#555555", style="italic")

# gain over stock (right axis): machine D line, other machines as points
ax2 = ax.twinx()
ratio = [f / s for f, s in zip(fork, stock)]
l3, = ax2.plot(share, ratio, "--^", color=C_GAIN, ms=4.5, lw=1.0, zorder=3)
ax2.plot(share[:2], ratio[:2], "^", mfc="white", mec=C_GAIN, ms=4.5, zorder=4)
A = [(36, 2.4), (35, 2.2), (45, 1.9), (55, 1.7), (58, 1.3), (87, 1.0)]
B = [(73, 2.03)]
l4 = ax2.scatter([x for x, _ in A], [y for _, y in A], s=22, marker="o", color=C_A, zorder=3)
l5 = ax2.scatter([x for x, _ in B], [y for _, y in B], s=26, marker="D", color=C_B, zorder=3)
ax2.axhline(1.0, color="#777777", lw=0.6, ls="--")
ax2.set_ylim(0.6, 2.8); ax2.set_ylabel("Gain over stock (x)", color=C_GAIN)
ax2.spines["right"].set_visible(True)
ax.legend([l1, l2, l3, l4, l5], ["fork (D, tokens/s)", "stock (D, tokens/s; open = modelled)", "gain, D (x)", "gain, machine A models (x)", "gain, laptop B (x)"],
          loc="upper center", frameon=False, ncol=2, handlelength=1.6, borderpad=0.3, labelspacing=0.3, columnspacing=1.2, bbox_to_anchor=(0.5, -0.13))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gain-trends.png")
fig.savefig(out); print(out)
