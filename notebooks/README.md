# Google Colab & Jupyter Notebooks

This directory contains Jupyter notebooks for interactive experimentation, visualization, and reproduction of the research on Google Colab or local JupyterLab environments.

## Available Notebooks

### 1. `recursive_learning_and_distillation_colab.ipynb` (End-to-End Master Notebook)
A complete, self-contained, publication-grade interactive research notebook designed to run smoothly on Google Colab with 1-click execution.

#### Contents & Workflow:
1. **Hardware & Environment Setup:** Automatic CUDA accelerator detection, Colab repo setup, reproducible seed initialization.
2. **Strict Data Protocol & Splitting:**
   - Fashion-MNIST download and stratified splitting (10% test, 10% val, 10% labeled training pool, 90% unlabeled training pool).
   - Zero-leakage verification checks.
   - Interactive class visualization grid.
3. **Model Architectures & Parameter Efficiency:**
   - Teacher (`SmallCNN`, ~212k parameters).
   - Student (`TinyCNN`, ~26k parameters, $8.15\times$ compression).
4. **Experiment 0: Supervised Baselines:**
   - Supervised Teacher training on 10% labeled pool.
   - Supervised Student scratch training on 10% labeled pool.
   - Learning curves, confusion matrices, and initial calibration curves.
5. **Experiment 1: Recursive Learning (Iterative Self-Training):**
   - Multi-round pseudo-labeling with confidence filtering ($\tau = 0.95$).
   - Interactive tracking of pseudo-label acceptance volume and diagnostic purity over rounds.
   - Confidence distribution sharpening histograms.
6. **Experiment 2: Knowledge Distillation:**
   - Distillation loss: $\alpha \mathcal{L}_{CE} + (1 - \alpha) T^2 \mathcal{D}_{KL}$.
   - Soft probability distribution visualization across temperatures $T \in [1, 2, 4, 8, 16]$.
   - Student distillation training.
   - 2D grid search heatmap over $(T, \alpha)$.
7. **Experiment 3: Combined Pipelines:**
   - Pipeline A: Recursive Teacher $\rightarrow$ Student Distillation.
   - Pipeline B: Distilled Student $\rightarrow$ Recursive Self-Training.
8. **Comprehensive Efficiency & Pareto Benchmarking:**
   - Real-world inference latency (ms/sample) and throughput (samples/sec).
   - Master comparative leaderboard (Accuracy, Macro F1, ECE, Parameters, Latency).
   - Accuracy vs. Latency Pareto frontier scatter plot.
   - Multi-model Reliability Diagrams (Expected Calibration Error).
9. **Scientific Findings & Recommendations.**

---

## How to Run in Google Colab

### Option A: Direct Upload to Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `notebooks/recursive_learning_and_distillation_colab.ipynb`.
3. Go to **Runtime $\rightarrow$ Change runtime type** and select **T4 GPU** (or CPU).
4. Run all cells (`Ctrl+F9` or `Runtime -> Run all`).

### Option B: Open via GitHub
1. Push your repository to GitHub.
2. In Google Colab, go to **File $\rightarrow$ Open notebook $\rightarrow$ GitHub tab**.
3. Paste your repository URL and select `notebooks/recursive_learning_and_distillation_colab.ipynb`.

---

## Execution Modes in the Notebook

The notebook includes a toggle cell right after the hardware check:
```python
FAST_DEMO = True   # Runs in ~2-3 minutes with 5 epochs & 2 self-training rounds
# OR
FAST_DEMO = False  # Full reproduction with 20 epochs & 5 self-training rounds
```
- **`FAST_DEMO = True` (Default):** Perfect for interactive demonstrations, grading, and quick verification in Google Colab without waiting for lengthy training.
- **`FAST_DEMO = False`:** Executes the full training schedule matching the experiment scripts.
