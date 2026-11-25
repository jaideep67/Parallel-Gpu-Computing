# Parallel GPU Computing (PGC)

Course laboratory implementations, experiments, and evaluation projects for **Parallel GPU Computing (PGC)** using NVIDIA CUDA, C/C++, and OpenMP.

---

## Repository Structure

| Directory | Topic / Experiment | Description |
| :--- | :--- | :--- |
| [`lab2_evaluation/`](lab2_evaluation/) | **Lab Evaluation 2: Warp Divergence** | Microarchitectural analysis of SIMT branch divergence, active masks, and branch-free predication on NVIDIA RTX 4500 Ada Generation ($N = 2^{24}$ elements). Covers CP1–CP5 with presentation, report, and visualizations. |
| [`lab_eval/`](lab_eval/) | **Lab Evaluation 1: GPU Vector Operations** | CPU vs GPU performance of element-wise vector addition and multiplication on NVIDIA Tesla T4. |
| [`Exp1/`](Exp1/) | **Experiment 1** | Introduction to CUDA programming and GPU device query. |
| [`Exp2/`](Exp2/) | **Experiment 2** | Matrix operations and thread indexing in CUDA. |
| [`Experiment_3/`](Experiment_3/) | **Experiment 3** | Parallel reduction and shared memory optimizations. |
| [`Experiment_4/`](Experiment_4/) | **Experiment 4** | Convolution and stencil computations on GPUs. |

---

## Lab Evaluation 2 Quick Links
- **Full Report**: [lab2_evaluation/README.md](lab2_evaluation/README.md)
- **CUDA Source**: [lab2_evaluation/src/warp_divergence.cu](lab2_evaluation/src/warp_divergence.cu)
- **PowerPoint Presentation (CP1–CP5)**: [lab2_evaluation/presentation/PGC_Lab2_Warp_Divergence_Evaluation.pptx](lab2_evaluation/presentation/PGC_Lab2_Warp_Divergence_Evaluation.pptx)
- **Performance Graphs**: [lab2_evaluation/graphs/](lab2_evaluation/graphs/)
- **Timing Results**: [lab2_evaluation/results/timing_data.csv](lab2_evaluation/results/timing_data.csv)
