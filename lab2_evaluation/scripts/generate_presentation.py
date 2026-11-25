#!/usr/bin/env python3
"""
Generates the PGC Lab 2 PowerPoint Presentation:
Theme: Warp Divergence in CUDA
Targeting all 5 Evaluation Checkpoints (CP1 to CP5)
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(BASE_DIR, 'graphs')
PPTX_PATH = os.path.join(BASE_DIR, 'presentation', 'PGC_Lab2_Warp_Divergence_Evaluation.pptx')

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color Palette: Deep Slate Blue, Emerald Green, Pure White, Accent Coral
NAVY = RGBColor(26, 37, 48)
LIGHT_BG = RGBColor(248, 249, 250)
GREEN = RGBColor(46, 204, 113)
DARK_BLUE = RGBColor(41, 128, 185)
WHITE = RGBColor(255, 255, 255)
TEXT_DARK = RGBColor(44, 62, 80)
MUTED = RGBColor(127, 140, 141)

blank_layout = prs.slide_layouts[6]

def add_header(slide, title_text, category_text="Parallel GPU Computing (PGC) · Lab Evaluation 2"):
    header_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
    header_box.fill.solid()
    header_box.fill.fore_color.rgb = NAVY
    header_box.line.color.rgb = NAVY
    
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.8)
    tf.margin_top = Inches(0.15)
    
    p0 = tf.paragraphs[0]
    p0.text = category_text.upper()
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = GREEN
    
    p1 = tf.add_paragraph()
    p1.text = title_text
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

# ------------------------------------------------------------------------------
# Slide 1: Title Slide
# ------------------------------------------------------------------------------
s1 = prs.slides.add_slide(blank_layout)
bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.color.rgb = NAVY

title_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(2.2))
tf = title_box.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "PARALLEL GPU COMPUTING LAB EVALUATION 2"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = GREEN

p2 = tf.add_paragraph()
p2.text = "Theme 7: CUDA Warp Divergence Analysis"
p2.font.size = Pt(36)
p2.font.bold = True
p2.font.color.rgb = WHITE

p3 = tf.add_paragraph()
p3.text = "Investigating SIMT Branch Serialization, Active Masks & Predication on NVIDIA RTX 4500 Ada"
p3.font.size = Pt(16)
p3.font.color.rgb = RGBColor(189, 195, 199)

# Team details table
team_shape = s1.shapes.add_table(5, 3, Inches(1.0), Inches(3.8), Inches(11.333), Inches(2.6))
tbl = team_shape.table
tbl.columns[0].width = Inches(1.5)
tbl.columns[1].width = Inches(5.5)
tbl.columns[2].width = Inches(4.333)

headers = ["Sl. No.", "Team Member Name", "USN / Roll Number"]
for c_idx, h in enumerate(headers):
    cell = tbl.cell(0, c_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = DARK_BLUE
    cp = cell.text_frame.paragraphs[0]
    cp.text = h
    cp.font.bold = True
    cp.font.size = Pt(13)
    cp.font.color.rgb = WHITE

members = [
    ("1", "Adarsh Walad", "01fe24bci"),
    ("2", "Nilaksha Nath", "01fe24bci087"),
    ("3", "Bachu venkata siva jaideep", "01fe24bci084"),
    ("4", "Prathamesh Eedigi", "01fe24bci006")
]

for r_idx, (num, name, usn) in enumerate(members, start=1):
    for c_idx, val in enumerate([num, name, usn]):
        cell = tbl.cell(r_idx, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(34, 49, 63)
        cp = cell.text_frame.paragraphs[0]
        cp.text = val
        cp.font.size = Pt(12)
        cp.font.color.rgb = WHITE

# ------------------------------------------------------------------------------
# Slide 2: CP1 - SIMT and Warps
# ------------------------------------------------------------------------------
s2 = prs.slides.add_slide(blank_layout)
add_header(s2, "Checkpoint 1: SIMT Architecture, Warps and Active Masks")

tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(5.8), Inches(5.6))
tf2 = tb2.text_frame
tf2.word_wrap = True

def add_bullet(tf, heading, body):
    p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
    p.text = heading + ": "
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = NAVY
    
    run = p.add_run()
    run.text = body
    run.font.bold = False
    run.font.size = Pt(12)
    run.font.color.rgb = TEXT_DARK
    p.space_after = Pt(10)

add_bullet(tf2, "Single Instruction, Multiple Threads (SIMT)", 
           "GPU hardware groups 32 parallel threads into an execution unit termed a Warp. All 32 threads in a warp issue and execute the exact same instruction simultaneously on SIMD execution units.")

add_bullet(tf2, "Warp Divergence Phenomenon", 
           "When threads in the same warp encounter a conditional branch (if-else) and evaluate differing boolean outcomes, the warp diverges. The hardware must serialize the execution of each divergent branch path.")

add_bullet(tf2, "Active Mask Mechanism", 
           "Hardware disables non-participating threads via a 32-bit Active Mask register. When Path A executes, threads taking Path B are masked off (idle). Then the warp executes Path B while Path A threads are disabled, effectively doubling latency.")

if os.path.exists(os.path.join(GRAPHS_DIR, '02_branch_efficiency_and_active_mask.png')):
    s2.shapes.add_picture(os.path.join(GRAPHS_DIR, '02_branch_efficiency_and_active_mask.png'), 
                          Inches(6.8), Inches(1.4), width=Inches(6.0))

# ------------------------------------------------------------------------------
# Slide 3: CP2 - Divergent vs Aligned Branch
# ------------------------------------------------------------------------------
s3 = prs.slides.add_slide(blank_layout)
add_header(s3, "Checkpoint 2: Divergent vs Warp-Aligned Branch Execution")

tb3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(5.8), Inches(5.6))
tf3 = tb3.text_frame
tf3.word_wrap = True

add_bullet(tf3, "Kernel 1 — Divergent Branch (threadIdx.x % 2)", 
           "Adjacent threads inside the same warp take opposite paths. Even-indexed lanes execute workA() while odd-indexed lanes execute workB(). Result: 50% lane utilization during each path, incurring serialization.")

add_bullet(tf3, "Kernel 2 — Warp-Aligned Branch ((threadIdx.x / 32) % 2)", 
           "The branch condition changes strictly on 32-thread boundaries. Every thread in Warp W executes the exact same path uniformly. Active mask remains 0xFFFFFFFF (100% lane utilization), eliminating intra-warp serialization.")

add_bullet(tf3, "Hardware Measurements on RTX 4500 Ada (20 runs)", 
           "• Divergent Branch Time: 0.3738 ms\n• Warp-Aligned Branch Time: 0.3955 ms\n• Divergent / Aligned Ratio: 0.945x\nOn modern Ada Lovelace, arithmetic predication and integer division differences affect sub-millisecond execution times.")

if os.path.exists(os.path.join(GRAPHS_DIR, '07_warp_lane_execution_timeline.png')):
    s3.shapes.add_picture(os.path.join(GRAPHS_DIR, '07_warp_lane_execution_timeline.png'), 
                          Inches(6.8), Inches(1.6), width=Inches(6.0))

# ------------------------------------------------------------------------------
# Slide 4: CP3 - Data-Dependent Branch & Optimization
# ------------------------------------------------------------------------------
s4 = prs.slides.add_slide(blank_layout)
add_header(s4, "Checkpoint 3: Data-Dependent Branching & Branch-Free Predication")

tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(5.8), Inches(5.6))
tf4 = tb4.text_frame
tf4.word_wrap = True

add_bullet(tf4, "Data-Dependent Branching Problem", 
           "In real-world workloads, condition outcomes depend on input values rather than thread IDs (e.g., input[i] > 0.5f). Since input data is typically non-uniform, threads within warps follow chaotic divergent trajectories.")

add_bullet(tf4, "Optimization 1: Sorting and Grouping Data", 
           "Rearranging input data prior to GPU processing (or spatially binning data) clusters similar values into coherent warp chunks, turning chaotic branches into warp-aligned executions.")

add_bullet(tf4, "Optimization 2: Branch-Free Code (Predication)", 
           "Both candidate results are computed unconditionally: float a = workA(x); float b = workB(x); out = (x > th) ? a : b;\nAvoids control flow instructions using hardware 'selp', but doubles the total compute work per thread (16 FLOPs vs 8 FLOPs).")

if os.path.exists(os.path.join(GRAPHS_DIR, '05_divergence_penalty_analysis.png')):
    s4.shapes.add_picture(os.path.join(GRAPHS_DIR, '05_divergence_penalty_analysis.png'), 
                          Inches(6.8), Inches(1.5), width=Inches(6.0))

# ------------------------------------------------------------------------------
# Slide 5: CP4 - Results Table & Benchmarks
# ------------------------------------------------------------------------------
s5 = prs.slides.add_slide(blank_layout)
add_header(s5, "Checkpoint 4: Experimental Results Table & Speedup Metrics")

# Add summary table
table_shape = s5.shapes.add_table(5, 5, Inches(0.8), Inches(1.4), Inches(6.0), Inches(2.6))
tbl5 = table_shape.table
tbl5.columns[0].width = Inches(2.0)
tbl5.columns[1].width = Inches(1.0)
tbl5.columns[2].width = Inches(1.0)
tbl5.columns[3].width = Inches(1.0)
tbl5.columns[4].width = Inches(1.0)

t_headers = ["Kernel Variant", "Avg Time", "Throughput", "Efficiency", "Ratio"]
for c_idx, h in enumerate(t_headers):
    cell = tbl5.cell(0, c_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = DARK_BLUE
    cp = cell.text_frame.paragraphs[0]
    cp.text = h
    cp.font.bold = True
    cp.font.size = Pt(10)
    cp.font.color.rgb = WHITE

rows = [
    ("Divergent Branch", "0.3738 ms", "44.88 G/s", "~50%", "1.000x"),
    ("Warp-Aligned Branch", "0.3955 ms", "42.42 G/s", "100%", "0.945x"),
    ("Data-Dependent", "0.3648 ms", "45.99 G/s", "~50%", "1.025x"),
    ("Branch-Free Predicated", "0.3952 ms", "42.45 G/s", "100%", "0.946x")
]

for r_idx, row in enumerate(rows, start=1):
    for c_idx, val in enumerate(row):
        cell = tbl5.cell(r_idx, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(240, 243, 244) if r_idx % 2 == 0 else WHITE
        cp = cell.text_frame.paragraphs[0]
        cp.text = val
        cp.font.size = Pt(9.5)
        cp.font.color.rgb = TEXT_DARK

# Key Insights
tb_ins = s5.shapes.add_textbox(Inches(0.8), Inches(4.3), Inches(6.0), Inches(2.7))
tf_ins = tb_ins.text_frame
tf_ins.word_wrap = True
add_bullet(tf_ins, "Hardware Configuration", "NVIDIA RTX 4500 Ada (SM 8.9), 16,777,216 elements, 256 threads/block (8 warps/block), 20 timed repetitions.")
add_bullet(tf_ins, "Core Findings", "Branch-free predication computes 16 FLOPs/thread, taking 0.3952 ms vs 0.3648 ms for data-dependent (0.923x ratio). Removing control-flow divergence does not always guarantee faster execution when redundant arithmetic is added.")

if os.path.exists(os.path.join(GRAPHS_DIR, '01_execution_time_comparison.png')):
    s5.shapes.add_picture(os.path.join(GRAPHS_DIR, '01_execution_time_comparison.png'), 
                          Inches(7.1), Inches(1.4), width=Inches(5.7))

# ------------------------------------------------------------------------------
# Slide 6: CP5 - Analysis & Industry Application
# ------------------------------------------------------------------------------
s6 = prs.slides.add_slide(blank_layout)
add_header(s6, "Checkpoint 5: Architectural Analysis & Industry Applications")

tb6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(6.0), Inches(5.6))
tf6 = tb6.text_frame
tf6.word_wrap = True

add_bullet(tf6, "Why Divergence Cost Depends on Path Count", 
           "If a warp encounters N divergent paths, total execution time scales as O(sum of all path durations). Every inactive thread lane represents wasted hardware execution cycles.")

add_bullet(tf6, "Industry Case 1: Real-time Ray Tracing & BVH Traversal", 
           "Rays traversing a Bounding Volume Hierarchy (BVH) reflect and refract in chaotic spatial directions. Adjacent threads quickly diverge, traversing different tree depths. Mitigated via ray sorting and RT Cores.")

add_bullet(tf6, "Industry Case 2: Physics Simulations & Particle Collisions", 
           "Particles colliding with non-uniform boundary geometry diverge in collision resolution kernels. Solved using spatial hashing and thread-coalesced particle bins.")

if os.path.exists(os.path.join(GRAPHS_DIR, '06_comprehensive_evaluation_dashboard.png')):
    s6.shapes.add_picture(os.path.join(GRAPHS_DIR, '06_comprehensive_evaluation_dashboard.png'), 
                          Inches(7.0), Inches(1.4), width=Inches(5.8))

prs.save(PPTX_PATH)
print(f"Presentation saved successfully at {PPTX_PATH}")
