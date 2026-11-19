"""
Experiment 3 - Analyze Synchronization Mechanisms in Parallel Programs
Performance-analysis figure generator.

Reads  : ../results/compare_sync_results.csv
         ../results/manual_sample_output.csv   (lab-manual sample, used only in Fig. 8)
Writes : ../graphs/*.png
         ../results/derived_metrics.csv

Usage  : python scripts/generate_graphs.py   (run from the Experiment_3 folder)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "graphs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data ----
# keep_default_na=False: the mechanism name "None" must stay a string, not become a missing value
df = pd.read_csv(os.path.join(RES, "compare_sync_results.csv"), keep_default_na=False)
ref = pd.read_csv(os.path.join(RES, "manual_sample_output.csv"), keep_default_na=False)

N = int(df.Expected.iloc[0])
MECH = ["None", "Critical", "Atomic", "Lock"]
THREADS = sorted(df.Threads.unique())
C = {"None": "#DC2626", "Critical": "#EA580C", "Atomic": "#16A34A", "Lock": "#7C3AED"}
MK = {"None": "o", "Critical": "s", "Atomic": "^", "Lock": "D"}
GREY, DARK = "#6B7280", "#1F2937"

t = df.pivot(index="Threads", columns="Mechanism", values="Time_s")[MECH]
cnt = df.pivot(index="Threads", columns="Mechanism", values="Final_Count")[MECH]
tref = ref.pivot(index="Threads", columns="Mechanism", values="Time_s")[MECH]
base = t.loc[1, "None"]                                   # 1-thread loop with no synchronization

# derived metrics (all from the measured table)
d = df.copy()
d["ns_per_increment"] = d.Time_s / N * 1e9
d["million_increments_per_s"] = N / d.Time_s / 1e6
d["lost_updates"] = N - d.Final_Count
d["lost_pct"] = d.lost_updates / N * 100
d["slowdown_vs_own_1_thread"] = d.apply(lambda r: r.Time_s / t.loc[1, r.Mechanism], axis=1)
d["times_slower_than_none_1_thread"] = d.Time_s / base
d["times_slower_than_atomic"] = d.apply(lambda r: r.Time_s / t.loc[r.Threads, "Atomic"], axis=1)
d.round(4).to_csv(os.path.join(RES, "derived_metrics.csv"), index=False)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.linestyle": "--", "grid.alpha": 0.35,
    "figure.dpi": 110, "savefig.dpi": 200,
    "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor="white")
    plt.close(fig)
    print("saved", name)


def box(ax, x, y, w, h, text, fc, ec="black", tc="black", fs=10, bold=False, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fc, edgecolor=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal")


# ------------------------------------------------- 01 mechanisms diagram --
fig, axes = plt.subplots(1, 4, figsize=(17, 6.4))
panels = [("None (race)", C["None"], "all threads at once", "lost updates", "#FEE2E2"),
          ("Critical / Lock", C["Critical"], "one thread at a time\n(others wait in a queue)", "correct, serial", "#FFEDD5"),
          ("Atomic", C["Atomic"], "one indivisible CPU\ninstruction (lock add)", "correct, no OS lock", "#DCFCE7"),
          ("Barrier", "#2563EB", "everyone waits until\nall threads arrive", "phases kept in order", "#DBEAFE")]
for ax, (title, col, how, res, fc) in zip(axes, panels):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, color=col, fontsize=14)
    for k in range(4):
        box(ax, 0.4 + k * 2.4, 8.2, 1.9, 1.0, f"T{k}", "#F3F4F6", fs=11, bold=True)
    if title.startswith("Barrier"):
        ax.plot([0.3, 9.7], [5.2, 5.2], color=col, lw=4)
        ax.text(5, 4.55, "#pragma omp barrier", ha="center", fontsize=10, color=col, fontweight="bold")
        for k, yy in enumerate([7.4, 6.4, 6.9, 5.9]):
            ax.annotate("", xy=(1.35 + k * 2.4, 5.3), xytext=(1.35 + k * 2.4, 8.15),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
        box(ax, 1.5, 2.6, 7, 1.3, "phase 2 starts only\nafter all reach the line", fc, ec=col, fs=10.5)
    else:
        box(ax, 2.5, 2.6, 5, 1.6, "shared\ncounter", fc, ec=col, fs=12, bold=True, lw=2)
        if title.startswith("Critical"):
            box(ax, 2.0, 4.9, 6, 0.9, "lock: 1 thread inside", "white", ec=col, fs=10, lw=1.8)
        for k in range(4):
            x = 1.35 + k * 2.4
            solid = not title.startswith("Critical") or k == 0
            ax.annotate("", xy=(5 + (k - 1.5) * 0.6, 4.25 if not title.startswith("Critical") else 5.85),
                        xytext=(x, 8.15),
                        arrowprops=dict(arrowstyle="-|>", color=col if solid else GREY, lw=1.8 if solid else 1.2,
                                        ls="-" if solid else ":"))
    ax.text(5, 1.6, how, ha="center", fontsize=10.5)
    ax.text(5, 0.5, res, ha="center", fontsize=11, fontweight="bold", color=col)
fig.suptitle("Fig. 1 - How Four Threads Reach One Shared Counter under Each Mechanism",
             fontsize=15, fontweight="bold")
save(fig, "01_synchronization_mechanisms.png")

# ------------------------------------------- 02 execution time vs threads --
fig, ax = plt.subplots(figsize=(10.5, 6))
# label offsets (points) chosen so the close Critical / Lock values do not overlap
off = {"None": (0, -18), "Critical": (0, -20), "Atomic": (0, -20), "Lock": (0, 11)}
for m in MECH:
    ls = "--" if m == "None" else "-"
    ax.plot(THREADS, t[m], marker=MK[m], ls=ls, color=C[m], lw=2.4, ms=9,
            label=m + ("  (wrong answer above 1 thread)" if m == "None" else ""))
    for p in THREADS:
        ax.annotate(f"{t.loc[p, m]:.4f}", (p, t.loc[p, m]), textcoords="offset points",
                    xytext=off[m], ha="center", fontsize=8.8, color=C[m], fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(0.0022, 0.9)
ax.set_xscale("log", base=2)
ax.set_xticks(THREADS)
ax.set_xticklabels(THREADS)
ax.set_xlim(0.85, 10)
ax.set_xlabel("Number of threads")
ax.set_ylabel("Execution time for 10,000,000 increments (s, log scale)")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.35))
ax.set_title("Fig. 2 - Execution Time vs Threads: Every Synchronized Version Gets Slower")
save(fig, "02_execution_time_vs_threads.png")

# ------------------------------------------------ 03 correctness ---------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
w = 0.2
for j, m in enumerate(MECH):
    xs = np.arange(len(THREADS)) + (j - 1.5) * w
    vals = cnt[m].to_numpy() / 1e6
    bars = a1.bar(xs, vals, w, color=C[m], edgecolor="black", label=m)
    for b, v in zip(bars, vals):
        if v < 9.99:
            a1.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}M", ha="center", fontsize=8.5,
                    color=C["None"], fontweight="bold")
a1.axhline(N / 1e6, color=DARK, ls="--", lw=1.4)
a1.text(3.55, N / 1e6 + 0.2, "expected 10M", ha="right", fontsize=9.5)
a1.set_xticks(range(len(THREADS)))
a1.set_xticklabels([f"{p} thread{'s' if p > 1 else ''}" for p in THREADS])
a1.set_ylim(0, 11.5)
a1.set_ylabel("Final counter value (millions)")
a1.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.1))
a1.set_title("(a) Final count: only 'None' is wrong", fontsize=12.5)

lost = d[d.Mechanism == "None"].set_index("Threads").lost_pct
a2.plot(THREADS, lost, "o-", color=C["None"], lw=2.4, ms=9, label="compare_sync 'None'")
for p in THREADS:
    a2.annotate(f"{lost[p]:.1f} %", (p, lost[p]), textcoords="offset points",
                xytext=(12, -16) if p == 4 else (-10, 10),
                fontsize=10, fontweight="bold", color=C["None"])
race_lost = (N - 3121112) / N * 100
a2.plot([4], [race_lost], "X", color=DARK, ms=12, label="race.c on its own (default threads)")
a2.annotate(f"race.c {race_lost:.1f} %", (4, race_lost), textcoords="offset points", xytext=(-95, 6), fontsize=10)
a2.set_xscale("log", base=2)
a2.set_xticks(THREADS)
a2.set_xticklabels(THREADS)
a2.set_ylim(-5, 100)
a2.set_xlabel("Number of threads")
a2.set_ylabel("Updates lost (%)")
a2.legend(loc="upper left")
a2.set_title("(b) Lost updates without synchronization", fontsize=12.5)
fig.suptitle("Fig. 3 - Correctness: the Race Condition Loses More Updates as Threads Increase",
             fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "03_correctness_and_lost_updates.png")

# -------------------------------------- 04 scalability (slowdown) -------
fig, ax = plt.subplots(figsize=(10.5, 6))
for m in MECH:
    s = t[m] / t.loc[1, m]
    ax.plot(THREADS, s, marker=MK[m], ls="--" if m == "None" else "-", color=C[m], lw=2.4, ms=9, label=m)
    ax.annotate(f"{s[8]:.2f}x", (8, s[8]), textcoords="offset points", xytext=(8, -3), fontsize=10,
                color=C[m], fontweight="bold")
ax.plot(THREADS, [1 / p for p in THREADS], ":", color=DARK, lw=1.8, label="ideal parallel scaling (time / p)")
ax.axhline(1, color=GREY, lw=1)
ax.text(5.5, 0.9, "same as 1 thread", fontsize=9, color=GREY)
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks(THREADS)
ax.set_xticklabels(THREADS)
ax.set_xlim(0.85, 11)
ax.set_xlabel("Number of threads")
ax.set_ylabel("Time / time with 1 thread (log scale)\nabove 1 = slower")
ax.legend(loc="lower left")
ax.set_title("Fig. 4 - Scalability: No Mechanism Speeds Up a Shared Counter")
save(fig, "04_scalability_relative_to_1_thread.png")

# ------------------------------------------ 05 cost per increment -------
fig, ax = plt.subplots(figsize=(12, 5.8))
ns = d.pivot(index="Threads", columns="Mechanism", values="ns_per_increment")[MECH]
for j, m in enumerate(MECH):
    xs = np.arange(len(THREADS)) + (j - 1.5) * w
    bars = ax.bar(xs, ns[m], w, color=C[m], edgecolor="black",
                  label="None (hatched: wrong result above 1 thread)" if m == "None" else m,
                  hatch="//" if m == "None" else None, alpha=0.75 if m == "None" else 1)
    for b, v in zip(bars, ns[m]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.8)
ax.set_xticks(range(len(THREADS)))
ax.set_xticklabels([f"{p} thread{'s' if p > 1 else ''}" for p in THREADS])
ax.set_ylabel("Nanoseconds per increment\n(total time / 10,000,000)")
ax.legend(loc="upper left", ncol=2)
ax.set_ylim(0, 62)
ax.set_title("Fig. 5 - Average Cost of One Increment")
save(fig, "05_cost_per_increment.png")

# ----------------------------------- 06 critical / lock vs atomic -------
fig, ax = plt.subplots(figsize=(10.5, 5.6))
for m in ["Critical", "Lock"]:
    r = t[m] / t["Atomic"]
    ax.plot(THREADS, r, marker=MK[m], color=C[m], lw=2.4, ms=9, label=f"{m} time / Atomic time")
    for p in THREADS:
        ax.annotate(f"{r[p]:.2f}x", (p, r[p]), textcoords="offset points",
                    xytext=(-12, 10 if m == "Lock" else -18), fontsize=10, color=C[m], fontweight="bold")
ax.axhline(1, color=C["Atomic"], ls="--", lw=1.5)
ax.text(1.05, 1.08, "Atomic", color=C["Atomic"], fontsize=10, fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(THREADS)
ax.set_xticklabels(THREADS)
ax.set_ylim(0, 5)
ax.set_xlabel("Number of threads")
ax.set_ylabel("How many times slower than Atomic")
ax.legend(loc="upper left")
ax.set_title("Fig. 6 - Atomic's Advantage Grows with Contention (1.7x -> 3.8-4.4x)")
save(fig, "06_critical_and_lock_vs_atomic.png")

# ------------------------------------------------ 07 barrier order -------
fig, ax = plt.subplots(figsize=(12, 5.2))
order = [("before", 0), ("before", 1), ("before", 2), ("before", 3),
         ("after", 3), ("after", 1), ("after", 0), ("after", 2)]
tc = {0: "#2563EB", 1: "#EA580C", 2: "#16A34A", 3: "#7C3AED"}
for i, (phase, th) in enumerate(order):
    ax.scatter(i + 1, th, s=420, color=tc[th], edgecolor="black", zorder=3,
               marker="o" if phase == "before" else "s")
    ax.text(i + 1, th + 0.33, phase, ha="center", fontsize=9.5)
ax.axvline(4.5, color=DARK, lw=3)
ax.text(4.5, 3.75, "#pragma omp barrier", ha="center", fontsize=11, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor=DARK))
ax.axvspan(0.5, 4.5, color="#DBEAFE", alpha=0.35)
ax.axvspan(4.5, 8.5, color="#DCFCE7", alpha=0.35)
ax.set_xlim(0.5, 8.5)
ax.set_ylim(-0.6, 4.2)
ax.set_xticks(range(1, 9))
ax.set_xlabel("Order in which the lines were printed (screenshot 01)")
ax.set_yticks(range(4))
ax.set_yticklabels([f"Thread {k}" for k in range(4)])
ax.grid(axis="x", visible=False)
ax.set_title("Fig. 7 - Barrier: All 4 'before' Lines Came Before Any 'after' Line")
save(fig, "07_barrier_output_order.png")

# ------------------------------------- 08 measured vs manual sample -----
fig, axes = plt.subplots(1, 4, figsize=(17, 5), sharey=True)
for ax, m in zip(axes, MECH):
    ax.plot(THREADS, tref[m], "o--", color=GREY, lw=2, ms=7, label="lab-manual sample")
    ax.plot(THREADS, t[m], marker=MK[m], color=C[m], lw=2.4, ms=9, label="measured (this VM)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(THREADS)
    ax.set_xticklabels(THREADS)
    ax.set_title(m, color=C[m], fontsize=13)
    ax.set_xlabel("Threads")
    ax.legend(loc="upper left", fontsize=9)
axes[0].set_ylabel("Time (s)")
fig.suptitle("Fig. 8 - Measured Times vs the Sample Output Printed in the Lab Manual (different machine)",
             fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "08_measured_vs_manual_sample.png")

# -------------------------------------------- 09 overhead heat map -------
fig, ax = plt.subplots(figsize=(10, 5.2))
hm = (t / base).T
im = ax.imshow(hm.to_numpy(), cmap="YlOrRd", aspect="auto")
for i, m in enumerate(MECH):
    for j, p in enumerate(THREADS):
        v = hm.loc[m, p]
        wrong = m == "None" and p > 1
        ax.text(j, i, f"{v:.1f}x" + ("\n(wrong)" if wrong else ""), ha="center", va="center",
                fontsize=11, fontweight="bold", color="white" if v > 90 else "black")
ax.set_xticks(range(len(THREADS)))
ax.set_xticklabels([f"{p} thread{'s' if p > 1 else ''}" for p in THREADS])
ax.set_yticks(range(len(MECH)))
ax.set_yticklabels(MECH)
ax.grid(False)
cb = fig.colorbar(im, ax=ax)
cb.set_label("times slower than 'None' with 1 thread")
ax.set_title(f"Fig. 9 - Cost of Each Mechanism Relative to the Plain 1-Thread Loop ({base:.4f} s)")
save(fig, "09_overhead_heatmap.png")

# --------------------------------------------------- 10 dashboard -------
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[0.72, 1.3], hspace=0.35, wspace=0.3)
kpis = [("Race condition", f"{lost[8]:.1f} %", "updates lost at 8 threads", C["None"]),
        ("Atomic at 8 threads", f"{t.loc[8, 'Atomic']:.4f} s", "fastest correct mechanism", C["Atomic"]),
        ("Critical at 8 threads", f"{t.loc[8, 'Critical'] / t.loc[8, 'Atomic']:.1f}x", "slower than Atomic", C["Critical"]),
        ("Lock at 8 threads", f"{t.loc[8, 'Lock'] / t.loc[8, 'Atomic']:.1f}x", "slower than Atomic", C["Lock"])]
for i, (title, big, small, col) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#F9FAFB", edgecolor=col, lw=2.5))
    ax.text(0.5, 0.77, title.upper(), ha="center", fontsize=10, color="#374151", transform=ax.transAxes)
    ax.text(0.5, 0.43, big, ha="center", fontsize=21, fontweight="bold", color=col, transform=ax.transAxes)
    ax.text(0.5, 0.14, small, ha="center", fontsize=10, color="#374151", transform=ax.transAxes)
ax = fig.add_subplot(gs[1, :])
ax.axis("off")
rows = [[f"{p}"] + [f"{t.loc[p, m]:.4f}" + ("  (X)" if m == "None" and cnt.loc[p, m] != N else "")
                    for m in MECH] for p in THREADS]
tbl = ax.table(cellText=rows, colLabels=["Threads", "None (s)", "Critical (s)", "Atomic (s)", "Lock (s)"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 2.3)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1F2937")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 1 and r > 1:
        cell.set_facecolor("#FEF2F2")
    elif c == 3:
        cell.set_facecolor("#F0FDF4")
ax.set_title("compare_sync: 10,000,000 shared increments   (X = wrong final count)", fontweight="bold")
fig.suptitle("Fig. 10 - Experiment 3 Summary: Correctness vs Speed of Synchronization",
             fontsize=15, fontweight="bold")
save(fig, "10_summary_dashboard.png")

print(d[["Mechanism", "Threads", "ns_per_increment", "lost_pct", "slowdown_vs_own_1_thread",
         "times_slower_than_atomic"]].round(2).to_string(index=False))
print("All figures written to", OUT)
