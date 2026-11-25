#!/usr/bin/env python3
"""
Generates the PGC Lab 2 PowerPoint Presentation:
Theme: Warp Divergence in CUDA
Format: 3-4 crisp points per checkpoint + Embedded Graph + 2-3 sentence Graph Takeaway
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(BASE_DIR, 'graphs')
PPTX_PATH = os.path.join(BASE_DIR, 'presentation', 'PGC_Lab2_Warp_Divergence_Evaluation.pptx')

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color Palette: Professional Slate Navy, Emerald Green, Clean Dark Text
NAVY = RGBColor(26, 37, 48)
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

def add_bullet(tf, heading, body):
    p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
    p.text = "• " + heading + ": "
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = NAVY
    
    run = p.add_run()
    run.text = body
    run.font.bold = False
    run.font.size = Pt(11.5)
    run.font.color.rgb = TEXT_DARK
    p.space_after = Pt(8)

def add_takeaway(slide, left, top, width, height, text):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(245, 247, 250)
    box.line.color.rgb = DARK_BLUE
    box.line.width = Pt(1.5)
    
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_right = Inches(0.2)
    tf.margin_top = Inches(0.12)
    
    p0 = tf.paragraphs[0]
    p0.text = "Graph Explanation (At a Glance):"
    p0.font.bold = True
    p0.font.size = Pt(11)
    p0.font.color.rgb = DARK_BLUE
    p0.space_after = Pt(4)
    
    p1 = tf.add_paragraph()
    p1.text = text
    p1.font.size = Pt(10)
    p1.font.color.rgb = TEXT_DARK

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
p3.text = "Simple Microarchitecture, Active Masks & Evaluation on NVIDIA RTX 4500 Ada (16.78M Elements)"
p3.font.size = Pt(15)
p3.font.color.rgb = RGBColor(189, 195, 199)

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
# Slide 2: Simple GPU Architecture
# ------------------------------------------------------------------------------
s2 = prs.slides.add_slide(blank_layout)
add_header(s2, "GPU Architecture: How CUDA Executes Code (At a Glance)")

tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.7), Inches(5.8))
tf2 = tb2.text_frame
tf2.word_wrap = True

add_bullet(tf2, "Host CPU to GPU Memory", 
           "The CPU allocates memory and copies the huge 16.78 million element array to the GPU's 24 GB GDDR6 memory (432 GB/s bandwidth) over the high-speed PCIe bus.")

add_bullet(tf2, "Streaming Multiprocessor (SM) & Warp Scheduling", 
           "The RTX 4500 Ada GPU contains 60 SMs. Each SM has dedicated warp schedulers that manage threads in groups of 32 called warps and issue instructions every clock cycle.")

add_bullet(tf2, "Lockstep Execution & Active Mask", 
           "32 physical CUDA cores execute one common instruction across the warp. A 32-bit hardware Active Mask register turns individual thread lanes ON (1) or OFF (0) when branches diverge.")

add_bullet(tf2, "Independent Thread Scheduling (ITS)", 
           "Modern Ada Lovelace architecture maintains per-thread Program Counters, allowing rapid interleaving of divergent branch paths without pipeline stalls.")

# ------------------------------------------------------------------------------
# Slide 3: CP1 - SIMT and Warps
# ------------------------------------------------------------------------------
s3 = prs.slides.add_slide(blank_layout)
add_header(s3, "Checkpoint 1: SIMT, 32-Thread Warps and Active Masks")

tb3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(5.8), Inches(4.2))
tf3 = tb3.text_frame
tf3.word_wrap = True

add_bullet(tf3, "Single Instruction, Multiple Threads (SIMT)", 
           "The GPU groups 32 threads into a single warp. All 32 threads share an instruction unit and execute instructions together in lockstep.")

add_bullet(tf3, "Branch Divergence Phenomenon", 
           "When threads in the same warp take different branches (e.g. if-else), the warp diverges because it cannot run two different instructions simultaneously.")

add_bullet(tf3, "Active Mask Serialization", 
           "Hardware uses a 32-bit Active Mask to disable threads taking the inactive path. Path A runs while Path B threads are masked (idle), cutting warp efficiency to 50%.")

add_bullet(tf3, "Reconvergence", 
           "Once both branch paths finish, the active mask restores to 0xFFFFFFFF (all 32 threads active) and the warp resumes unified execution.")

if os.path.exists(os.path.join(GRAPHS_DIR, '02_branch_efficiency_and_active_mask.png')):
    s3.shapes.add_picture(os.path.join(GRAPHS_DIR, '02_branch_efficiency_and_active_mask.png'), 
                          Inches(6.8), Inches(1.3), width=Inches(5.8))

add_takeaway(s3, Inches(6.8), Inches(5.1), Inches(5.8), Inches(1.8),
             "1. Shows active lanes across warp-aligned, divergent, and branch-free kernels.\n"
             "2. In divergent branching (middle), half the lanes are active (green) and half are idle (red).\n"
             "3. The aligned branch keeps all 32 lanes active at 100% efficiency.")

# ------------------------------------------------------------------------------
# Slide 4: CP2 - Divergent vs Aligned Branch
# ------------------------------------------------------------------------------
s4 = prs.slides.add_slide(blank_layout)
add_header(s4, "Checkpoint 2: Divergent vs. Warp-Aligned Branch Execution")

tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(5.8), Inches(4.2))
tf4 = tb4.text_frame
tf4.word_wrap = True

add_bullet(tf4, "Kernel 1 — Divergent Branch (i % 2)", 
           "Adjacent threads take opposite paths. Even threads execute workA while odd threads execute workB, forcing every warp to serialize both paths.")

add_bullet(tf4, "Kernel 2 — Warp-Aligned Branch ((i / 32) % 2)", 
           "Branch condition changes only at 32-thread boundaries. Every thread in a warp takes the exact same path uniformly, eliminating intra-warp divergence.")

add_bullet(tf4, "Execution Timeline Impact", 
           "The divergent warp takes two clock phases (2T) to complete both paths, whereas the aligned warp completes all work in a single clock phase (T).")

add_bullet(tf4, "Measured Times on RTX 4500 Ada", 
           "Divergent branch took 0.3738 ms, while warp-aligned branch took 0.3955 ms (ratio: 0.945x). Ada Lovelace handles short loops smoothly via predication.")

if os.path.exists(os.path.join(GRAPHS_DIR, '07_warp_lane_execution_timeline.png')):
    s4.shapes.add_picture(os.path.join(GRAPHS_DIR, '07_warp_lane_execution_timeline.png'), 
                          Inches(6.8), Inches(1.4), width=Inches(5.8))

add_takeaway(s4, Inches(6.8), Inches(5.1), Inches(5.8), Inches(1.8),
             "1. Contrasts the execution timeline of divergent versus aligned warps.\n"
             "2. Divergent warps must run Step 1 for even lanes and Step 2 for odd lanes.\n"
             "3. Aligned warps complete in one phase because all 32 lanes stay synchronized.")

# ------------------------------------------------------------------------------
# Slide 5: CP3 - Data-Dependent Branch & Optimization
# ------------------------------------------------------------------------------
s5 = prs.slides.add_slide(blank_layout)
add_header(s5, "Checkpoint 3: Data-Dependent Branching & Branch-Free Predication")

tb5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(5.8), Inches(4.2))
tf5 = tb5.text_frame
tf5.word_wrap = True

add_bullet(tf5, "The Real-World Problem", 
           "Real data is rarely pre-arranged by thread ID. When branches depend on runtime input data (input[i] > 0.5), threads experience unpredictable divergence.")

add_bullet(tf5, "Fix 1: Data Sorting and Grouping", 
           "Sorting or grouping input data before GPU processing clusters similar values together, restoring warp alignment naturally.")

add_bullet(tf5, "Fix 2: Branch-Free Predication", 
           "Writing output = (input[i] > th) ? a : b uses hardware conditional select (selp) instructions, completely eliminating control flow branch jumps.")

add_bullet(tf5, "The Work Inflation Trade-off", 
           "Branch-free code forces every thread to compute both candidate math paths (16 FLOPs instead of 8), increasing execution time from 0.3648 ms to 0.3952 ms.")

if os.path.exists(os.path.join(GRAPHS_DIR, '05_divergence_penalty_analysis.png')):
    s5.shapes.add_picture(os.path.join(GRAPHS_DIR, '05_divergence_penalty_analysis.png'), 
                          Inches(6.8), Inches(1.3), width=Inches(5.8))

add_takeaway(s5, Inches(6.8), Inches(5.1), Inches(5.8), Inches(1.8),
             "1. Compares computed FLOPs per thread against active warp lanes.\n"
             "2. Branch-free achieves 100% active lanes but doubles arithmetic work to 16 FLOPs.\n"
             "3. Proves removing divergence is counterproductive if it doubles math work.")

# ------------------------------------------------------------------------------
# Slide 6: CP4 - Results Table & Benchmarks
# ------------------------------------------------------------------------------
s6 = prs.slides.add_slide(blank_layout)
add_header(s6, "Checkpoint 4: Experimental Results Table & Performance Dashboard")

table_shape = s6.shapes.add_table(5, 5, Inches(0.8), Inches(1.3), Inches(5.8), Inches(2.4))
tbl6 = table_shape.table
tbl6.columns[0].width = Inches(2.0)
tbl6.columns[1].width = Inches(0.95)
tbl6.columns[2].width = Inches(0.95)
tbl6.columns[3].width = Inches(0.95)
tbl6.columns[4].width = Inches(0.95)

t_headers = ["Kernel Variant", "Avg Time", "Throughput", "Efficiency", "Ratio"]
for c_idx, h in enumerate(t_headers):
    cell = tbl6.cell(0, c_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = DARK_BLUE
    cp = cell.text_frame.paragraphs[0]
    cp.text = h
    cp.font.bold = True
    cp.font.size = Pt(9.5)
    cp.font.color.rgb = WHITE

rows = [
    ("Divergent Branch", "0.3738 ms", "44.88 G/s", "~50%", "1.000x"),
    ("Warp-Aligned Branch", "0.3955 ms", "42.42 G/s", "100%", "0.945x"),
    ("Data-Dependent", "0.3648 ms", "45.99 G/s", "~50%", "1.025x"),
    ("Branch-Free Predicated", "0.3952 ms", "42.45 G/s", "100%", "0.946x")
]

for r_idx, row in enumerate(rows, start=1):
    for c_idx, val in enumerate(row):
        cell = tbl6.cell(r_idx, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(240, 243, 244) if r_idx % 2 == 0 else WHITE
        cp = cell.text_frame.paragraphs[0]
        cp.text = val
        cp.font.size = Pt(9)
        cp.font.color.rgb = TEXT_DARK

tb_ins = s6.shapes.add_textbox(Inches(0.8), Inches(3.9), Inches(5.8), Inches(3.0))
tf_ins = tb_ins.text_frame
tf_ins.word_wrap = True
add_bullet(tf_ins, "Test Setup", "NVIDIA RTX 4500 Ada (SM 8.9), 16,777,216 elements, 256 threads/block, 20 timed repetitions.")
add_bullet(tf_ins, "Fastest Execution", "Data-dependent branch ran fastest at 0.3648 ms (45.99 G-elements/s throughput).")
add_bullet(tf_ins, "Core Conclusion", "All kernels ran in 0.36 ms to 0.40 ms, demonstrating massive parallel throughput.")

if os.path.exists(os.path.join(GRAPHS_DIR, '06_comprehensive_evaluation_dashboard.png')):
    s6.shapes.add_picture(os.path.join(GRAPHS_DIR, '06_comprehensive_evaluation_dashboard.png'), 
                          Inches(6.8), Inches(1.3), width=Inches(5.8))

add_takeaway(s6, Inches(6.8), Inches(5.1), Inches(5.8), Inches(1.8),
             "1. Executive dashboard combining time, throughput, speedup, and hardware specs.\n"
             "2. All kernels achieve over 42,000 million elements per second throughput.\n"
             "3. Gives evaluators a complete performance summary in one single view.")

# ------------------------------------------------------------------------------
# Slide 7: CP5 - Analysis & Industry Application
# ------------------------------------------------------------------------------
s7 = prs.slides.add_slide(blank_layout)
add_header(s7, "Checkpoint 5: Multi-Path Penalty & Real-World Industry Uses")

tb7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(5.8), Inches(4.2))
tf7 = tb7.text_frame
tf7.word_wrap = True

add_bullet(tf7, "Why Divergence Cost Multiplies", 
           "If a warp branches into K paths, total time is the sum of all path times (T = T1 + T2 + ... + TK). If all 32 threads take different paths, GPU efficiency drops to just 3.1%.")

add_bullet(tf7, "Industry Case 1: Real-Time Ray Tracing", 
           "Secondary rays bounce in random directions, causing threads to search different branches of a 3D BVH tree. Solved via ray binning and dedicated hardware RT Cores.")

add_bullet(tf7, "Industry Case 2: Physics & Fluid Simulations", 
           "Particles hitting obstacles need complex collision math, while open-space particles do simple math. Solved by spatial grid hashing.")

add_bullet(tf7, "Key Takeaway for GPU Coders", 
           "To achieve peak performance, always organize memory and data so adjacent threads perform the same computations together.")

if os.path.exists(os.path.join(GRAPHS_DIR, '04_throughput_and_giga_elements.png')):
    s7.shapes.add_picture(os.path.join(GRAPHS_DIR, '04_throughput_and_giga_elements.png'), 
                          Inches(6.8), Inches(1.3), width=Inches(5.8))

add_takeaway(s7, Inches(6.8), Inches(5.1), Inches(5.8), Inches(1.8),
             "1. Plots effective processing throughput in million elements per second.\n"
             "2. All kernels process over 42,400 M elements/s, peaking at 45,990 M elements/s.\n"
             "3. Shows how subtle branch and arithmetic differences impact real GPU throughput.")

prs.save(PPTX_PATH)
print(f"Presentation saved successfully at {PPTX_PATH}")
