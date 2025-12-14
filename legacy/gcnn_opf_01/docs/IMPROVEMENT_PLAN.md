# IMPROVEMENT_PLAN.md

This plan focuses on upgrading the model architecture and training strategy to match the performance reported in the Gao et al. paper, specifically targeting Probabilistic Accuracy > 90%.

## 0. 🛡️ Setup
- [x] **Create New Branch**
  - Create and switch to a new branch (e.g., `feature/capacity-and-training`) to preserve the current working baseline.

## 1. 🏗️ Increase Model Capacity
**Objective:** Expand the neural network's capacity to capture high-frequency voltage variations and precise constraints, aligning with the 1000-neuron architecture in the paper.

- [x] **Modify `gcnn_opf_01/config_model_01.py`**
  - Locate `ModelConfig`.
  - Update `neurons_fc`:
    ```python
    neurons_fc: int = 1000  # Increased from 128 to match paper (or 512 if memory constrained)
    ```

## 2. 🔄 Implement Two-Stage Training Strategy
**Objective:** Replicate the "M4 to M5" strategy. First, learn the general mapping (Supervised); then, snap to the physics manifold (Physics-Informed).

- [x] **Modify `gcnn_opf_01/train.py`**
  - Refactor `main()` to execute two distinct training phases sequentially.
  - **Phase 1 (Pre-training):**
    - Train for **25 epochs**.
    - Set `kappa = 0.0` (Pure Supervised Loss).
    - Learning Rate: `1e-3`.
    - Save checkpoint: `results/model_phase1.pth`.
  - **Phase 2 (Fine-tuning):**
    - Load weights from `results/model_phase1.pth`.
    - Train for **25 epochs**.
    - Set `kappa = 1.0` (Strong Physics Penalty).
    - **Lower** Learning Rate: `1e-4` (critical for fine-tuning).
    - Save final model: `results/final_model_refined.pth`.

## 3. 🎛️ Re-Verify Batch Size (Quick Tune)
**Objective:** Ensure the optimal batch size hasn't shifted due to the much larger model parameters.

- [x] **Run Tuning Script**
  - Execute `python gcnn_opf_01/tune_batch_size.py --batch_sizes 6 16 32`.
  - *Decision:* If 6 remains the best, proceed. If a larger batch size (e.g., 16) is now stable and comparable, prefer it for faster training speed.

## 4. 🚀 Execution & Evaluation
- [x] **Run Training**
  - Execute `python gcnn_opf_01/train.py` with the new two-stage logic.
- [x] **Evaluate Performance**
  - Run `python gcnn_opf_01/evaluate.py`.
  - **Key Metric Check:**
    - Probabilistic Accuracy ($P_{PG}$ and $P_{VG}$) with thresholds 1 MW / 0.001 p.u.
    - Check if $P$ values have improved from <40% towards the >90% target.

## 5. ✅ Final Results (2025-11-26)
- **Goal Achieved:** Yes
- **Probabilistic Accuracy ($P_{PG}$):** 98.42%
- **Voltage Accuracy ($P_{VG}$):** 100.00%
- **Optimal Batch Size:** 24
- **Model Capacity:** 1000 neurons
- **Note:** The initial low accuracy (<40%) was primarily due to a normalization error in the evaluation script. Correcting this revealed that even the baseline model performed well (>96%), but the 1000-neuron model improved precision further.
    - Check Gen 1 (Slack) MSE to see if the larger capacity helped the underestimation issue.