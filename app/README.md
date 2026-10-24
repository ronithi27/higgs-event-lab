# HiggsLens — Interactive Scientific Dashboard

**Owner:** Ronit Singh (Product UI / Streamlit Dashboard).

HiggsLens provides an interactive, scientific workbench for inspecting machine-learning experiments on the simulated UCI HIGGS dataset (Baldi et al., 2014). It visualizes held-out evaluation curves, compares classical baselines against compact neural networks across feature groups, and enables interactive operating-threshold exploration.

---

## 1. Quickstart & Launch Instructions

### Prerequisites
Ensure the virtual environment is activated and project dependencies are installed:

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install repository in editable mode
pip install -e .
```

### Launch the Dashboard
Run the Streamlit application from the repository root:

```bash
streamlit run app/main.py
```

The application will launch locally at `http://localhost:8501`.

---

## 2. Navigation & Application Architecture

HiggsLens is organized into six focused scientific sections:

1. **Overview:** Project objectives, dataset parameters, artifact registry discovery status, and verified reproduction commands.
2. **Physics Explorer:** Standard Model electroweak symmetry breaking, the exotic-Higgs benchmark decay kinematics, and explicit project boundaries.
3. **Dataset Observatory:** Searchable 28-feature dictionary, interactive signal vs background distribution explorer, cross-feature Pearson correlation matrix, and empirical class balance observatory.
4. **Experiment Comparison:** Model comparison leaderboard, multi-group representation learning analysis (`low_level` vs `high_level` vs `all`), and training runtime vs performance trade-off scatter plot.
5. **Validation:** Receiver Operating Characteristic (ROC) curves, Precision-Recall (PR) curves, calibration reliability diagrams, genuine MLP training loss/AUC curves, and the interactive threshold operating-point workbench.
6. **Methods and Research:** The `higgs-split-v1` leakage-free protocol, neural and linear model configurations, academic citations, and research limitations.

---

## 3. Artifact Discovery & Schema Contract

HiggsLens dynamically discovers run folders beneath `outputs/` (or a custom path entered in the sidebar). A run folder is classified as usable only when it contains both required files complying with `higgs-experiment-v1`:

- **`metrics.json`**: Metadata dictionary including schema version, model architecture, feature group, split seed, training configuration, runtime, and validation/test metrics.
- **`test_predictions.csv`**: Held-out test predictions containing `row_index`, `class_label` ($y \in \{0, 1\}$), and predicted class-1 probability `score` ($\in [0.0, 1.0]$).

### Honest Empty State Policy
If `outputs/` is missing or contains no valid runs, HiggsLens displays an honest empty state with verified CLI commands to reproduce runs. It never simulates fake benchmark metrics or trains heavy models on dashboard startup.

---

## 4. Reproducing Experiment Runs

To populate the dashboard with genuine benchmark runs, execute the verified CLI scripts:

```bash
# 1. Extract a reproducible subset (e.g. 250,000 events)
python -m higgs_event_lab.data \
    --source data/raw/HIGGS.csv.gz \
    --output data/processed/higgs_sample_250k.csv.gz \
    --rows 250000

# 2. Train Logistic Regression baselines across all 3 feature groups
python -m higgs_event_lab.classical \
    --data data/processed/higgs_sample_250k.csv.gz \
    --all-groups \
    --output-dir outputs

# 3. Train Compact MLP neural networks across all 3 feature groups
python -m higgs_event_lab.neural \
    --data data/processed/higgs_sample_250k.csv.gz \
    --all-groups \
    --output-dir outputs
```

---

## 5. Automated Tests

Targeted unit tests for the dashboard loader, schema validation, and threshold calculations are located in `tests/unit/`:

```bash
# Run dashboard unit tests
pytest tests/unit/test_app_loader.py tests/unit/test_app_metrics.py -v

# Run entire repository test suite
pytest
```
