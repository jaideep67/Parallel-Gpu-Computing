"""
Experiment 4 - Cache Coherence (MSI, MESI, MOESI) and NUMA Memory Access Performance
Performance-analysis figure generator.

Reads  : ../results/coherence_sim_results.csv
         ../results/coherence_state_trace.csv
         ../results/false_sharing_runs.csv
         ../results/memory_sweep_runs.csv
Writes : ../graphs/*.png
         ../results/summary_metrics.csv

Usage  : python scripts/generate_graphs.py   (run from the Experiment_4 folder)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "graphs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data ----
sim = pd.read_csv(os.path.join(RES, "coherence_sim_results.csv"))
trace = pd.read_csv(os.path.join(RES, "coherence_state_trace.csv"))
fs = pd.read_csv(os.path.join(RES, "false_sharing_runs.csv"))
sw = pd.read_csv(os.path.join(RES, "memory_sweep_runs.csv"))

PROTO = ["MSI", "MESI", "MOESI"]
C_P = {"MSI": "#2563EB", "MESI": "#16A34A", "MOESI": "#7C3AED"}
C_ADJ, C_PAD = "#DC2626", "#16A34A"
C_1T, C_2T, C_LOCAL = "#2563EB", "#EA580C", "#7C3AED"
GREY, DARK = "#6B7280", "#1F2937"
STATE_C = {"M": "#DC2626", "O": "#F59E0B", "E": "#16A34A", "S": "#2563EB", "I": "#E5E7EB"}

sim["Bus_transactions"] = sim.BusRd + sim.BusRdX + sim.BusUpgr
adj = sim[sim.Layout == "adjacent"].set_index("Protocol").loc[PROTO]
pad = sim[sim.Layout == "padded"].set_index("Protocol").loc[PROTO]

# false sharing: standalone 2-thread runs; the 3 repeats give the median the manual asks for
st = fs[fs.Program == "false_sharing"]
st_adj = st[st.Layout == "Adjacent"].Time_s.to_numpy()
st_pad = st[st.Layout == "Padded"].Time_s.to_numpy()
rep = st[st.Group.str.startswith("repeat")]
med_adj = rep[rep.Layout == "Adjacent"].Time_s.median()
med_pad = rep[rep.Layout == "Padded"].Time_s.median()
cn = fs[fs.Program == "compare_numa"]

# memory sweep: 3 repeats (default placement) give the medians
LINES = (256 << 20) // 64 * 4                      # cache lines touched in 4 passes
BYTES = LINES * 64                                 # = 1 GiB moved through the caches
srep = sw[sw.Group.str.startswith("repeat")]
med_1 = srep[srep.Threads == 1].Time_s.median()
med_2 = srep[srep.Threads == 2].Time_s.median()
loc_1 = sw[(sw.Placement != "default") & (sw.Threads == 1)].Time_s.iloc[0]
loc_2 = sw[(sw.Placement != "default") & (sw.Threads == 2)].Time_s.iloc[0]

summary = pd.DataFrame([
    ("false sharing, adjacent, 2 threads (median of 3)", med_adj, "s"),
    ("false sharing, padded, 2 threads (median of 3)", med_pad, "s"),
    ("adjacent / padded (cost of false sharing)", med_adj / med_pad, "x"),
    ("memory sweep, 1 thread, default (median of 3)", med_1, "s"),
    ("memory sweep, 2 threads, default (median of 3)", med_2, "s"),
    ("memory sweep speedup 1 -> 2 threads", med_1 / med_2, "x"),
    ("effective bandwidth, 1 thread", BYTES / med_1 / 1e9, "GB/s"),
    ("effective bandwidth, 2 threads", BYTES / med_2 / 1e9, "GB/s"),
    ("memory sweep, 1 thread, numactl node 0 (local)", loc_1, "s"),
    ("memory sweep, 2 threads, numactl node 0 (local)", loc_2, "s"),
    ("memory sweep, remote node (--membind=1)", np.nan, "not possible: 1 NUMA node"),
], columns=["Metric", "Value", "Unit"])
summary.round(4).to_csv(os.path.join(RES, "summary_metrics.csv"), index=False)

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


def box(ax, x, y, w, h, text, fc, ec="black", tc="black", fs_=10, bold=False, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=fc, edgecolor=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, color=tc,
                fontweight="bold" if bold else "normal")


# ----------------------------------------- 01 coherence + false sharing ---
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1, 1.15]})
for ax in (a1, a2):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
for k, x in enumerate([0.6, 5.4]):
    box(a1, x, 7.6, 4, 1.4, f"Core {k}\n(thread {k})", "#DBEAFE", fs_=11, bold=True)
    box(a1, x, 5.4, 4, 1.5, f"private cache {k}\nline A: copy", "#F3F4F6", fs_=10.5)
    a1.annotate("", xy=(x + 2, 6.95), xytext=(x + 2, 7.55), arrowprops=dict(arrowstyle="<->", lw=1.5))
    a1.annotate("", xy=(x + 2, 4.25), xytext=(x + 2, 5.35), arrowprops=dict(arrowstyle="<->", lw=1.5))
a1.plot([0.4, 9.6], [4.1, 4.1], color=DARK, lw=5)
a1.text(0.4, 3.55, "shared bus: every cache\nSNOOPS (watches) it", ha="left", va="center",
        fontsize=10.5, fontweight="bold")
box(a1, 5.4, 1.0, 4, 1.5, "main memory (RAM)", "#E5E7EB", fs_=11, bold=True)
a1.annotate("", xy=(7.4, 2.55), xytext=(7.4, 4.0), arrowprops=dict(arrowstyle="<->", lw=1.5))
a1.set_title("(a) Two cores, two private caches, one shared bus", fontsize=12.5)

for row, (lab, layout, col) in enumerate([("Adjacent (stride 1)", "adj", C_ADJ),
                                           ("Padded (stride 8 = 64 bytes)", "pad", C_PAD)]):
    y0 = 6.3 if row == 0 else 1.6
    a2.text(0.2, y0 + 2.35, lab, fontsize=12, fontweight="bold", color=col)
    nlines = 1 if layout == "adj" else 2
    for L in range(nlines):
        x0 = 0.2 + L * 4.9
        a2.add_patch(plt.Rectangle((x0, y0), 4.6, 1.5, facecolor="white", edgecolor=col, lw=2.2))
        a2.text(x0 + 2.3, y0 - 0.45, f"cache line {'AB'[L]} (64 bytes = 8 longs)", ha="center", fontsize=9.5)
        for s in range(8):
            if layout == "adj":
                txt, fc = ("cnt[0]\nthr 0", "#DBEAFE") if s == 0 else (("cnt[1]\nthr 1", "#FFEDD5") if s == 1 else ("", "#F9FAFB"))
            else:
                txt, fc = (("cnt[0]\nthr 0", "#DBEAFE") if L == 0 else ("cnt[8]\nthr 1", "#FFEDD5")) if s == 0 else ("pad", "#F9FAFB")
            a2.add_patch(plt.Rectangle((x0 + s * 0.575, y0), 0.575, 1.5, facecolor=fc, edgecolor=GREY, lw=0.6))
            a2.text(x0 + s * 0.575 + 0.29, y0 + 0.75, txt, ha="center", va="center", fontsize=7.5)
    note = ("both counters in ONE line -> every write by one core\ninvalidates the other core's copy (false sharing)"
            if layout == "adj" else "each counter has its OWN line -> each core keeps\nits line in state M, no traffic between cores")
    a2.text(5.0 if layout == "adj" else 5.0, y0 + 1.55 + 0.25 if layout == "pad" else y0 + 0.1, "", fontsize=1)
    a2.text(5.2 if layout == "adj" else 0.2, y0 + 0.75 if layout == "adj" else y0 - 1.3, note,
            fontsize=10, color=col, va="center")
a2.set_title("(b) The two-counter example: where cnt[0] and cnt[1] live in memory", fontsize=12.5)
fig.suptitle("Fig. 1 - Cache Coherence Works on Whole 64-Byte Lines, Not on Variables",
             fontsize=15, fontweight="bold")
save(fig, "01_coherence_and_false_sharing_concept.png")

# ------------------------------------------------ 02 state trace ---------
fig, ax = plt.subplots(figsize=(15, 6.2))
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-0.6, 3 * 2.4)
ax.axis("off")
for pi, p in enumerate(PROTO):
    ybase = (2 - pi) * 2.4
    ax.text(-0.45, ybase + 0.95, p, fontsize=14, fontweight="bold", color=C_P[p], ha="left", va="center")
    for si, r in trace.iterrows():
        s0, s1 = r[p].split("/")
        for ci, s in enumerate([s0, s1]):
            yy = ybase + (1.05 if ci == 0 else 0.15)
            ax.add_patch(plt.Rectangle((si + 0.12, yy), 0.76, 0.75, facecolor=STATE_C[s], edgecolor="black", lw=0.8))
            ax.text(si + 0.5, yy + 0.37, s, ha="center", va="center", fontsize=13, fontweight="bold",
                    color="black" if s in "IE" else "white")
    ax.text(8.15, ybase + 1.42, "core 0", fontsize=9, color=GREY)
    ax.text(8.15, ybase + 0.52, "core 1", fontsize=9, color=GREY)
for si, r in trace.iterrows():
    who, verb = r.Operation.split(" reads ")[0] if " reads " in r.Operation else r.Operation.split(" writes ")[0], \
        ("R" if " reads " in r.Operation else "W")
    ax.text(si + 0.5, 3 * 2.4 - 0.25, f"{si + 1}. {who.replace('core ', 'C')} {verb}", ha="center",
            fontsize=10, fontweight="bold")
ax.legend(handles=[Patch(facecolor=STATE_C[s], edgecolor="black", label=lab) for s, lab in
                   [("M", "M Modified"), ("O", "O Owned"), ("E", "E Exclusive"), ("S", "S Shared"), ("I", "I Invalid")]],
          loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=10.5)
ax.set_title("Fig. 2 - State of Cache Line A in Each Core, First 2 Iterations (adjacent layout)", pad=12)
save(fig, "02_protocol_state_trace.png")

# ---------------------------------------- 03 coherence events adjacent ---
events = ["BusRd", "BusUpgr", "Invalidations", "Writebacks", "CacheToCache", "Silent_E_to_M"]
labels = ["Bus reads\n(BusRd)", "Upgrades\n(BusUpgr)", "Invalidations", "Write-backs\nto memory",
          "Cache-to-cache\ntransfers", "Silent E->M\nwrites"]
fig, ax = plt.subplots(figsize=(13.5, 5.8))
w = 0.26
for j, p in enumerate(PROTO):
    xs = np.arange(len(events)) + (j - 1) * w
    vals = adj.loc[p, events].to_numpy()
    bars = ax.bar(xs, vals, w, color=C_P[p], edgecolor="black", label=p)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 30, f"{v:,}", ha="center", fontsize=8.8, rotation=0)
ax.set_xticks(range(len(events)))
ax.set_xticklabels(labels)
ax.set_ylim(0, 2400)
ax.set_ylabel("Count for 1,000 iterations\n(4,000 memory operations)")
ax.legend(loc="upper right")
ax.set_title("Fig. 3 - Coherence Events with False Sharing (adjacent layout): MOESI Replaces Write-backs")
save(fig, "03_coherence_events_adjacent.png")

# -------------------------------------- 04 adjacent vs padded traffic ---
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6))
x = np.arange(len(PROTO))
b1 = a1.bar(x - 0.2, adj.Bus_transactions, 0.4, color=C_ADJ, edgecolor="black", label="adjacent (same line)")
b2 = a1.bar(x + 0.2, pad.Bus_transactions, 0.4, color=C_PAD, edgecolor="black", label="padded (own lines)")
for b, v in list(zip(b1, adj.Bus_transactions)) + list(zip(b2, pad.Bus_transactions)):
    a1.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
a1.set_yscale("log")
a1.set_ylim(1, 20000)
a1.set_xticks(x)
a1.set_xticklabels(PROTO)
a1.set_ylabel("Bus transactions (log scale)\nBusRd + BusRdX + BusUpgr")
a1.legend(loc="upper right")
a1.set_title("(a) Bus traffic: about 1,000-2,000x less when padded", fontsize=12.5)

b1 = a2.bar(x - 0.2, adj.Invalidations, 0.4, color=C_ADJ, edgecolor="black", label="adjacent")
b2 = a2.bar(x + 0.2, pad.Invalidations, 0.4, color=C_PAD, edgecolor="black", label="padded")
for b, v in list(zip(b1, adj.Invalidations)) + list(zip(b2, pad.Invalidations)):
    a2.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
a2.set_xticks(x)
a2.set_xticklabels(PROTO)
a2.set_ylim(0, 2400)
a2.set_ylabel("Copies invalidated")
a2.legend(loc="upper right")
a2.set_title("(b) Invalidations: 1,999 vs 0 for every protocol", fontsize=12.5)
fig.suptitle("Fig. 4 - Padding Removes Almost All Coherence Traffic (simulation, 1,000 iterations)",
             fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "04_adjacent_vs_padded_traffic.png")

# ------------------------------------ 05 false sharing, 2 threads -------
fig, ax = plt.subplots(figsize=(12, 5.8))
grp = ["single run", "repeat 1", "repeat 2", "repeat 3"]
x = np.arange(len(grp))
ba = ax.bar(x - 0.2, st_adj, 0.4, color=C_ADJ, edgecolor="black", label="Adjacent (false sharing)")
bp = ax.bar(x + 0.2, st_pad, 0.4, color=C_PAD, edgecolor="black", label="Padded")
for i in range(len(grp)):
    ax.text(x[i] - 0.2, st_adj[i] + 0.0005, f"{st_adj[i]:.4f}", ha="center", fontsize=9)
    ax.text(x[i] + 0.2, st_pad[i] + 0.0005, f"{st_pad[i]:.4f}", ha="center", fontsize=9)
    ax.text(x[i], max(st_adj[i], st_pad[i]) + 0.0035, f"{st_adj[i] / st_pad[i]:.2f}x", ha="center",
            fontsize=11, fontweight="bold", color=DARK)
ax.axhline(med_adj, color=C_ADJ, ls="--", lw=1.3)
ax.axhline(med_pad, color=C_PAD, ls="--", lw=1.3)
ax.text(-0.45, med_adj + 0.0004, f"median of repeats {med_adj:.4f} s", color=C_ADJ, ha="left", fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1))
ax.text(-0.45, med_pad + 0.0004, f"median of repeats {med_pad:.4f} s", color=C_PAD, ha="left", fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1))
ax.set_xticks(x)
ax.set_xticklabels(grp)
ax.set_ylim(0, 0.039)
ax.set_ylabel("Time for 100,000,000 increments (s)")
ax.legend(loc="upper right")
ax.set_title("Fig. 5 - False Sharing with 2 Threads: Padded Was Faster in All 4 Pairs")
save(fig, "05_false_sharing_two_threads.png")

# --------------------------- 06 compare_numa: 1-thread control ---------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
for ax, run in zip((a1, a2), sorted(cn.Run.unique())):
    d = cn[cn.Run == run]
    for k, th in enumerate([1, 2]):
        va = d[(d.Threads == th) & (d.Layout == "Adjacent")].Time_s.iloc[0]
        vp = d[(d.Threads == th) & (d.Layout == "Padded")].Time_s.iloc[0]
        b1 = ax.bar(k - 0.2, va, 0.4, color=C_ADJ, edgecolor="black", label="Adjacent" if k == 0 else None)
        b2 = ax.bar(k + 0.2, vp, 0.4, color=C_PAD, edgecolor="black", label="Padded" if k == 0 else None)
        ax.text(k - 0.2, va + 0.0005, f"{va:.4f}", ha="center", fontsize=9.5)
        ax.text(k + 0.2, vp + 0.0005, f"{vp:.4f}", ha="center", fontsize=9.5)
        ax.text(k, max(va, vp) + 0.0035, f"ratio {va / vp:.2f}x", ha="center", fontsize=11, fontweight="bold")
    ax.axvspan(-0.5, 0.5, color="#FEF3C7", alpha=0.5, zorder=0)
    ax.text(0, 0.0372, "1 thread = control\n(no other core, no false sharing possible)", ha="center", fontsize=9.5,
            color="#92400E")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1 thread", "2 threads"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(d.Group.iloc[0], fontsize=12.5)
    ax.legend(loc="upper right")
a1.set_ylabel("Time (s)")
a1.set_ylim(0, 0.042)
fig.suptitle("Fig. 6 - compare_numa: the Adjacent/Padded Gap Appears Even with 1 Thread",
             fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "06_compare_numa_false_sharing_with_control.png")

# --------------------------------------------- 07 memory sweep -----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.3, 1]})
mk = {"single run (first after compiling)": ("X", 130), "single run": ("o", 80)}
for th, col in [(1, C_1T), (2, C_2T)]:
    d = sw[sw.Threads == th]
    for _, r in d.iterrows():
        xpos = th + (0.18 if r.Placement != "default" else (-0.18 if r.Program == "compare_numa" else 0))
        if r.Placement != "default":
            a1.scatter(xpos, r.Time_s, s=140, marker="D", color=C_LOCAL, edgecolor="black", zorder=3)
        elif r.Program == "compare_numa":
            a1.scatter(xpos, r.Time_s, s=90, marker="s", color=col, edgecolor="black", alpha=0.6, zorder=3)
        else:
            m, s = mk.get(r.Group, ("o", 80))
            a1.scatter(xpos, r.Time_s, s=s, marker=m, color=col, edgecolor="black", zorder=3)
    med = med_1 if th == 1 else med_2
    a1.plot([th - 0.3, th + 0.3], [med, med], color=col, lw=3)
    a1.text(th + 0.32, med, f"median {med:.4f}", va="center", fontsize=9.5, color=col, fontweight="bold")
a1.set_xticks([1, 2])
a1.set_xticklabels(["1 thread", "2 threads"])
a1.set_xlim(0.5, 2.85)
a1.set_ylabel("Time for 4 passes over 256 MB (s)")
a1.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=GREY, markeredgecolor="black", label="numa_sweep (default)"),
                   plt.Line2D([], [], marker="X", ls="", color=GREY, markeredgecolor="black", ms=10, label="first run after compiling"),
                   plt.Line2D([], [], marker="s", ls="", color=GREY, markeredgecolor="black", label="compare_numa"),
                   plt.Line2D([], [], marker="D", ls="", color=C_LOCAL, markeredgecolor="black", label="numactl node 0 (local)")],
          loc="upper right", fontsize=9.5)
a1.set_title("(a) All 14 sweep timings", fontsize=12.5)

bw = [BYTES / med_1 / 1e9, BYTES / med_2 / 1e9]
bars = a2.bar([0, 1], bw, 0.55, color=[C_1T, C_2T], edgecolor="black")
for b, v in zip(bars, bw):
    a2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f} GB/s", ha="center", fontweight="bold")
a2.set_xticks([0, 1])
a2.set_xticklabels(["1 thread", "2 threads"])
a2.set_ylim(0, 27)
a2.set_ylabel("Effective bandwidth (GB/s)\n1 GiB of cache lines / median time")
a2.text(0.5, 24.5, f"speedup only {med_1 / med_2:.2f}x\n(ideal 2x)", ha="center", fontsize=11, fontweight="bold")
a2.set_title("(b) Bandwidth from the medians", fontsize=12.5)
fig.suptitle("Fig. 7 - Memory Sweep: a Second Thread Adds Only ~13 % (memory-bound)",
             fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "07_memory_sweep.png")

# ---------------------------------------------- 08 NUMA topology ---------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.4))
for ax in (a1, a2):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
box(a1, 1.5, 1.2, 7, 7.3, "", "#F0FDF4", ec=C_PAD, lw=2.5)
a1.text(5, 7.9, "NUMA node 0", ha="center", fontsize=13, fontweight="bold", color="#166534")
box(a1, 2.2, 5.6, 2.6, 1.4, "CPU 0\n(socket 0)", "#DBEAFE", fs_=11, bold=True)
box(a1, 5.2, 5.6, 2.6, 1.4, "CPU 1\n(socket 1)", "#DBEAFE", fs_=11, bold=True)
box(a1, 2.2, 2.0, 5.6, 2.2, "memory: 5,360 MB\n(free 134 MB at that moment)", "#E5E7EB", fs_=11)
a1.annotate("", xy=(3.5, 4.25), xytext=(3.5, 5.55), arrowprops=dict(arrowstyle="<->", lw=1.5))
a1.annotate("", xy=(6.5, 4.25), xytext=(6.5, 5.55), arrowprops=dict(arrowstyle="<->", lw=1.5))
a1.text(5, 0.5, "numactl --hardware: available: 1 nodes (0), distance 10 (local only)", ha="center", fontsize=10.5)
a1.set_title("(a) THIS VM (measured): one node -> every access is local", fontsize=12.5, color="#166534")

for k, x0 in enumerate([0.3, 5.3]):
    box(a2, x0, 1.8, 4.4, 6.4, "", "#F9FAFB", ec=GREY, lw=2)
    a2.text(x0 + 2.2, 7.6, f"node {k}", ha="center", fontsize=12.5, fontweight="bold", color=GREY)
    box(a2, x0 + 0.4, 5.3, 3.6, 1.4, f"CPUs {2 * k}, {2 * k + 1}", "#E5E7EB", fs_=11)
    box(a2, x0 + 0.4, 2.4, 3.6, 1.6, f"local memory {k}", "#E5E7EB", fs_=11)
a2.plot([4.7, 5.3], [5.0, 5.0], color=DARK, lw=4)
a2.text(5, 8.6, "interconnect between the nodes", ha="center", fontsize=9.5)
a2.annotate("", xy=(5, 5.1), xytext=(5, 8.45), arrowprops=dict(arrowstyle="-", lw=1, color=GREY, ls=":"))
a2.annotate("", xy=(5.9, 3.1), xytext=(2.6, 5.25),
            arrowprops=dict(arrowstyle="-|>", lw=2, color="#DC2626", ls="--"))
a2.text(5, 1.1, "red arrow: remote access (CPU on node 0, memory on node 1: --membind=1)",
        color="#DC2626", fontsize=10, ha="center")
a2.text(5, 0.3, "the manual's target: 4 vCPUs in 2 virtual nodes (.vmx settings) - NOT available here",
        ha="center", fontsize=10.5, color=GREY)
a2.set_title("(b) What a 2-node layout would look like (not measured)", fontsize=12.5, color=GREY)
fig.suptitle("Fig. 8 - NUMA Topology: Why the Remote-Memory Test Could Not Be Run on This VM",
             fontsize=15, fontweight="bold")
save(fig, "08_numa_topology.png")

# --------------------------------------------------- 09 dashboard -------
fig = plt.figure(figsize=(14, 8.2))
gs = fig.add_gridspec(2, 4, height_ratios=[0.7, 1.35], hspace=0.32, wspace=0.3)
kpis = [("Bus traffic (MSI)", f"{adj.Bus_transactions['MSI']:,} -> {pad.Bus_transactions['MSI']}",
         "adjacent -> padded, simulated", C_P["MSI"]),
        ("MOESI write-backs", f"{adj.Writebacks['MOESI']}", f"vs {adj.Writebacks['MSI']:,} (MSI / MESI)", C_P["MOESI"]),
        ("False sharing, 2 threads", f"{med_adj / med_pad:.2f}x", "adjacent / padded (medians)", C_ADJ),
        ("Sweep speedup, 2 threads", f"{med_1 / med_2:.2f}x", f"{BYTES / med_2 / 1e9:.1f} GB/s with 2 threads", C_2T)]
for i, (title, big, small, col) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#F9FAFB", edgecolor=col, lw=2.5))
    ax.text(0.5, 0.77, title.upper(), ha="center", fontsize=9.3, color="#374151", transform=ax.transAxes)
    ax.text(0.5, 0.43, big, ha="center", fontsize=19, fontweight="bold", color=col, transform=ax.transAxes)
    ax.text(0.5, 0.14, small, ha="center", fontsize=9.5, color="#374151", transform=ax.transAxes)
ax = fig.add_subplot(gs[1, :])
ax.axis("off")
rows = [["adjacent, 2 threads (false_sharing)", f"{med_adj:.4f}", "median of 3 repeats"],
        ["padded, 2 threads (false_sharing)", f"{med_pad:.4f}", "median of 3 repeats"],
        ["memory sweep, 1 thread, default", f"{med_1:.4f}", "median of 3 repeats"],
        ["memory sweep, 2 threads, default", f"{med_2:.4f}", "median of 3 repeats"],
        ["memory sweep, LOCAL node 0, 1 thread", f"{loc_1:.4f}", "single run (numactl)"],
        ["memory sweep, LOCAL node 0, 2 threads", f"{loc_2:.4f}", "single run (numactl)"],
        ["memory sweep, REMOTE node", "Not possible", "VM shows only 1 NUMA node"]]
tbl = ax.table(cellText=rows, colLabels=["Result to record (manual 9.3)", "Time (s)", "Note"],
               loc="center", cellLoc="center", colWidths=[0.45, 0.2, 0.35])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1F2937")
        cell.set_text_props(color="white", fontweight="bold")
    elif r == 7:
        cell.set_facecolor("#F3F4F6")
fig.suptitle("Fig. 9 - Experiment 4 Summary: Coherence Traffic, False Sharing and Memory Access",
             fontsize=15, fontweight="bold")
save(fig, "09_summary_dashboard.png")

print(summary.round(4).to_string(index=False))
print("All figures written to", OUT)
