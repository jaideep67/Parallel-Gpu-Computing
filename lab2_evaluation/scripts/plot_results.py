#!/usr/bin/env python3
"""
Performance Visualization Script for PGC Lab 2: CUDA Warp Divergence
Target Architecture: NVIDIA RTX 4500 Ada Generation (SM 8.9)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Resolve directories relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(BASE_DIR, 'graphs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'axes.edgecolor': '#2C3E50',
    'axes.linewidth': 1.2,
    'grid.color': '#BDC3C7',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'figure.autolayout': True
})

# Load benchmark data
csv_path = os.path.join(RESULTS_DIR, 'timing_data.csv')
df = pd.read_csv(csv_path)

colors = ['#E74C3C', '#2ECC71', '#3498DB', '#9B59B6']

# ==============================================================================
# Graph 1: Execution Time Comparison
# ==============================================================================
plt.figure(figsize=(9, 5.5), dpi=300)
bars = plt.bar(df['kernel_name'], df['avg_time_ms'], color=colors, width=0.55, edgecolor='#1A252F', linewidth=1.2)
plt.title('CUDA Kernel Execution Time (Average over 20 runs)\nNVIDIA RTX 4500 Ada Generation · 16,777,216 Elements', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
plt.ylim(0, 0.48)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f'{yval:.4f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(rotation=15, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, '01_execution_time_comparison.png'))
plt.close()

# ==============================================================================
# Graph 2: Active Mask & SIMT Warp Lane Efficiency
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
warp_lanes = np.arange(32)

# Case 1: Divergent (alternating)
divergent_mask = [1 if i % 2 == 0 else 0 for i in warp_lanes]
# Case 2: Aligned (all active for same branch)
aligned_mask = [1 for _ in warp_lanes]
# Case 3: Branch-Free (predicated - all lanes calculate both)
branchfree_mask = [1 for _ in warp_lanes]

y_positions = [2, 1, 0]
labels = ['Warp-Aligned Branch\n(Full Warp 100% Efficiency)', 
          'Divergent Branch (i % 2)\n(50% Active, 50% Masked)', 
          'Branch-Free Predication\n(100% Active, 2x Operations)']

for y_idx, (y_pos, pattern) in enumerate(zip(y_positions, [aligned_mask, divergent_mask, branchfree_mask])):
    for lane in range(32):
        if y_pos == 1:
            color = '#2ECC71' if pattern[lane] == 1 else '#E74C3C'
            status = 'Active' if pattern[lane] == 1 else 'Masked'
        elif y_pos == 2:
            color = '#3498DB'
        else:
            color = '#9B59B6'
        rect = patches.Rectangle((lane * 0.9, y_pos - 0.25), 0.75, 0.5, facecolor=color, edgecolor='#2C3E50', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(lane * 0.9 + 0.375, y_pos, str(lane), ha='center', va='center', fontsize=6.5, color='white' if y_pos != 0 else 'white', fontweight='bold')

ax.set_xlim(-0.5, 32 * 0.9 + 0.5)
ax.set_ylim(-0.8, 2.8)
ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=9.5, fontweight='bold')
ax.set_xlabel('Warp Lane Index (0 to 31)', fontsize=11, fontweight='bold')
ax.set_title('SIMT Warp Execution & Active Mask State (32 Threads/Warp)', fontsize=12, fontweight='bold', pad=15)
ax.grid(False)

# Custom legend
legend_patches = [
    patches.Patch(facecolor='#2ECC71', edgecolor='#2C3E50', label='Active Thread Lane (Path A)'),
    patches.Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='Masked / Idle Lane (Serialized)'),
    patches.Patch(facecolor='#3498DB', edgecolor='#2C3E50', label='Uniform Active Warp'),
    patches.Patch(facecolor='#9B59B6', edgecolor='#2C3E50', label='Predicated Lane (Computes Both A & B)')
]
ax.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1.0, -0.28), ncol=2, fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, '02_branch_efficiency_and_active_mask.png'), bbox_inches='tight')
plt.close()

# ==============================================================================
# Graph 3: Relative Speedup Ratios
# ==============================================================================
plt.figure(figsize=(8.5, 5), dpi=300)
ratios = [1.0, 0.3738 / 0.3955, 0.3738 / 0.3648, 0.3738 / 0.3952]
ratio_labels = ['Divergent (Baseline 1.0x)', 'Warp-Aligned (0.945x)', 'Data-Dependent (1.025x)', 'Branch-Free (0.946x)']
ratio_colors = ['#7F8C8D', '#27AE60', '#2980B9', '#8E44AD']

bars = plt.bar(ratio_labels, ratios, color=ratio_colors, width=0.5, edgecolor='#1A252F', linewidth=1.2)
plt.axhline(1.0, color='#C0392B', linestyle='--', linewidth=1.5, label='Baseline (Divergent)')
plt.ylabel('Relative Speedup vs Divergent Kernel', fontsize=11, fontweight='bold')
plt.title('Relative Execution Performance Ratio\n(Values > 1.0 indicate faster execution than divergent baseline)', fontsize=11, fontweight='bold', pad=12)
plt.ylim(0, 1.25)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(rotation=10, ha='right', fontsize=9.5)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, '03_relative_speedup_ratios.png'))
plt.close()

# ==============================================================================
# Graph 4: Throughput (Million Elements Processed Per Second)
# ==============================================================================
plt.figure(figsize=(9, 5), dpi=300)
throughputs = (16777216 / (df['avg_time_ms'] / 1000.0)) / 1e6
bars = plt.bar(df['kernel_name'], throughputs, color=colors, width=0.55, edgecolor='#1A252F', linewidth=1.2)
plt.title('Effective Processing Throughput (Mega-Elements / sec)\nN = 16.77M FP32 Elements · NVIDIA RTX 4500 Ada', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Throughput (Million Elements/s)', fontsize=11, fontweight='bold')
plt.ylim(0, 52000)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 900, f'{yval:,.1f} M/s', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.xticks(rotation=15, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, '04_throughput_and_giga_elements.png'))
plt.close()

# ==============================================================================
# Graph 5: Divergence vs Arithmetic Penalty Analysis (Work Inflation)
# ==============================================================================
plt.figure(figsize=(8.5, 5), dpi=300)
categories = ['Divergent\n(i % 2)', 'Aligned\n((i/32) % 2)', 'Data-Dep\n(threshold)', 'Branch-Free\n(both paths)']
arithmetic_ops = [8, 8, 8, 16] # 8 FLOPs per thread vs 16 FLOPs for both workA & workB
active_mask_pct = [50, 100, 50, 100]

x = np.arange(len(categories))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=300)
ax2 = ax1.twinx()

rects1 = ax1.bar(x - width/2, arithmetic_ops, width, label='FLOPs Computed / Thread', color='#E67E22', edgecolor='#1A252F')
rects2 = ax2.bar(x + width/2, active_mask_pct, width, label='Branch Active Mask (%)', color='#1ABC9C', edgecolor='#1A252F')

ax1.set_ylabel('Arithmetic Ops Per Thread (FLOPs)', fontsize=10, fontweight='bold', color='#D35400')
ax2.set_ylabel('Active Warp Lanes (%)', fontsize=10, fontweight='bold', color='#16A085')
ax1.set_title('Instruction Inflation vs Branch Masking Trade-off', fontsize=12, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
ax1.set_ylim(0, 20)
ax2.set_ylim(0, 120)

for r in rects1:
    ax1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f'{int(r.get_height())}', ha='center', fontsize=9, fontweight='bold')
for r in rects2:
    ax2.text(r.get_x() + r.get_width()/2, r.get_height() + 2, f'{int(r.get_height())}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, '05_divergence_penalty_analysis.png'))
plt.close()

# ==============================================================================
# Graph 6: Executive Performance Dashboard
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

# Subplot (0,0): Execution Time
axs[0, 0].bar(df['kernel_name'], df['avg_time_ms'], color=colors, edgecolor='#1A252F')
axs[0, 0].set_title('Execution Time (ms)', fontsize=11, fontweight='bold')
axs[0, 0].set_ylabel('Time (ms)', fontsize=10)
axs[0, 0].set_ylim(0, 0.48)
axs[0, 0].tick_params(axis='x', rotation=20)
for bar in axs[0, 0].patches:
    axs[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, f'{bar.get_height():.4f} ms', ha='center', fontsize=8.5, fontweight='bold')

# Subplot (0,1): Throughput
axs[0, 1].bar(df['kernel_name'], throughputs, color=colors, edgecolor='#1A252F')
axs[0, 1].set_title('Throughput (Million Elements / sec)', fontsize=11, fontweight='bold')
axs[0, 1].set_ylabel('M Elements/s', fontsize=10)
axs[0, 1].tick_params(axis='x', rotation=20)
for bar in axs[0, 1].patches:
    axs[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800, f'{bar.get_height():,.0f}', ha='center', fontsize=8.5, fontweight='bold')

# Subplot (1,0): Speedup Ratios
axs[1, 0].bar(ratio_labels, ratios, color=ratio_colors, edgecolor='#1A252F')
axs[1, 0].axhline(1.0, color='red', linestyle='--', label='1.0x Baseline')
axs[1, 0].set_title('Speedup Ratio vs Divergent', fontsize=11, fontweight='bold')
axs[1, 0].set_ylabel('Ratio', fontsize=10)
axs[1, 0].tick_params(axis='x', rotation=20)
axs[1, 0].set_ylim(0, 1.25)
for bar in axs[1, 0].patches:
    if bar.get_height() > 0:
        axs[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height():.3f}x', ha='center', fontsize=8.5, fontweight='bold')

# Subplot (1,1): Architectural Specs & Findings Summary Table
axs[1, 1].axis('off')
table_data = [
    ['GPU Model', 'NVIDIA RTX 4500 Ada'],
    ['Compute Arch', 'SM 8.9 (Ada Lovelace)'],
    ['Data Elements', '16,777,216 (2^24)'],
    ['Threads/Block', '256 (8 Warps/Block)'],
    ['Timed Iterations', '20 runs (CUDA Events)'],
    ['Divergent Time', '0.3738 ms'],
    ['Aligned Time', '0.3955 ms'],
    ['Data-Dep Time', '0.3648 ms'],
    ['Branch-Free Time', '0.3952 ms']
]
tbl = axs[1, 1].table(cellText=table_data, colLabels=['Parameter', 'Observed Value'], loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.1, 1.35)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#34495E')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#F8F9F9' if row % 2 == 0 else '#FFFFFF')

fig.suptitle('PGC Lab 2: CUDA Warp Divergence Evaluation Dashboard', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, '06_comprehensive_evaluation_dashboard.png'))
plt.close()

# ==============================================================================
# Graph 7: Warp Lane Execution Timeline (Serialization Diagram)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

# Divergent Warp timeline
ax.barh(y=1, width=1, left=0, height=0.4, color='#E74C3C', edgecolor='black', label='Step 1: Even Lanes active (workA), Odd lanes masked')
ax.barh(y=1, width=1, left=1, height=0.4, color='#E67E22', edgecolor='black', label='Step 2: Odd Lanes active (workB), Even lanes masked')

# Aligned Warp timeline
ax.barh(y=0, width=1, left=0, height=0.4, color='#2ECC71', edgecolor='black', label='All 32 Lanes active simultaneously (workA or workB)')

ax.set_yticks([0, 1])
ax.set_yticklabels(['Warp-Aligned Warp\n(Single Phase)', 'Divergent Warp\n(Serialized Execution)'], fontsize=10, fontweight='bold')
ax.set_xlabel('Instruction Execution Phase / Clock Time', fontsize=10, fontweight='bold')
ax.set_title('SIMT Hardware Execution: Why Divergence Causes Serialization', fontsize=12, fontweight='bold', pad=15)
ax.set_xlim(0, 2.2)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax.set_xticklabels(['0', 'T/2', 'T (Aligned Complete)', '1.5 T', '2T (Divergent Complete)'])
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.legend(loc='upper right', fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, '07_warp_lane_execution_timeline.png'))
plt.close()

print("All 7 graphs successfully generated in graphs/ directory.")
