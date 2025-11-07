# Feature Specification: Comparative Analysis of CNN vs ViT on BloodMNIST

**Feature Branch**: `001-compare-cnn-vit-bloodmnist`  
**Created**: 2025-11-05  
**Status**: Draft  
**Input**: User description: "Evaluate CNNs vs Vision Transformers on BloodMNIST with shared pipeline, quantitative metrics, interpretability visuals, and reproducible setup."

## User Scenarios & Testing (mandatory)

### User Story 1 - Train and Evaluate Both Architectures (Priority: P1)

A researcher trains one CNN baseline and one ViT baseline on BloodMNIST using the same preprocessing, split policy, and evaluation protocol, and receives a side-by-side metrics comparison.

**Why this priority**: Delivers the core value: a fair, apples-to-apples comparison to inform model choice.

**Independent Test**: Run a single command/workflow to produce trained models plus a metrics summary table comparing CNN vs ViT.

**Acceptance Scenarios**:

1. Given the official dataset splits and a fixed random seed, When the training workflow is executed for CNN and ViT, Then both models complete training and persist artifacts (weights, logs, metrics).
2. Given trained models, When the evaluation workflow runs on the designated split(s), Then it outputs Accuracy, Precision, Recall, F1, and AUC for each model and a comparison table.

---

### User Story 2 - Interpretability Visualizations (Priority: P2)

An ML practitioner generates interpretability visualizations (e.g., Grad-CAM for CNN and attention maps for ViT) for selected samples to compare what each model focuses on.

**Why this priority**: Supports model trust and domain insight beyond metrics alone.

**Independent Test**: Run a visualization workflow that, given trained models and sample images, produces saved visual overlays and a lightweight gallery.

**Acceptance Scenarios**:

1. Given trained models and a list of images, When the visualization workflow runs, Then it outputs heatmaps/attention overlays to disk for at least N samples per class (N >= 10 by default).
2. Given generated overlays, When a reviewer opens the artifacts, Then each image clearly shows the model’s areas of focus without obscuring diagnostic content.

---

### User Story 3 - Reproducible End-to-End Run (Priority: P3)

An auditor or teammate can reproduce the full comparison from scratch (data to report) following documented steps and obtain comparable results within a small tolerance.

**Why this priority**: Ensures credibility and future extensibility.

**Independent Test**: Follow documented steps to reproduce training and evaluation; outputs match within a predefined tolerance.

**Acceptance Scenarios**:

1. Given a clean environment, When the documented setup and single-command workflows are executed, Then the pipeline completes without manual code edits.
2. Given defined random seeds and the same dataset revision, When the pipeline is rerun, Then key metrics vary by <= 1.0 percentage point absolute.

---

### Edge Cases

- Dataset fetch or integrity check fails (network, checksum, or license unavailability).
- Class imbalance skews metrics; per-class metrics must be reported alongside macro/micro averages.
- Non-determinism from data shuffling/augmentations; runs must set seeds and document tolerances.
- Hardware without acceleration; provide a reduced compute mode and clear expectations.
- Early stopping or time budget triggers; partial training still yields valid comparisons.
- Visualization overlays are illegible; enforce minimum contrast and file-size limits.

## Requirements (mandatory)

### Functional Requirements

- FR-001: Provide a standardized data pipeline that uses the official BloodMNIST splits and documents any preprocessing applied.
- FR-002: Train one baseline CNN and one baseline ViT using identical preprocessing, split policy, and evaluation protocol.
- FR-003: Report quantitative metrics per model: Accuracy, Precision, Recall, F1 (macro and micro where applicable), and ROC-AUC (macro and micro where applicable).
- FR-004: Persist artifacts for each run: trained weights, configuration (hyperparameters, seeds), logs, confusion matrices, and a metrics CSV.
- FR-005: Generate interpretability visuals: class-activation overlays for the CNN and attention-based maps for the ViT, covering at least 10 samples per class by default.
- FR-006: Produce a comparative summary (table and brief narrative) describing performance, efficiency (e.g., epochs/time), and interpretability observations.
- FR-007: Provide a single-command workflow (or CLI/script) to reproduce training, evaluation, and artifact generation end-to-end.
- FR-008: Ensure reproducibility by fixing random seeds, recording dataset revision, and documenting expected variance thresholds (<= 1.0 pp absolute on primary metrics).
- FR-009: Include a lightweight results notebook for exploration and figure generation (metrics tables, curves, and overlays reference).
- FR-010: Publish a concise report artifact summarizing goals, methods, results, and takeaways for non-technical stakeholders.
- FR-011: Offer a reduced-compute mode that completes within a constrained time budget while preserving fair comparison logic.
- FR-012: Interactive inference demo is deferred to a later iteration and excluded from the scope of this feature.

### Key Entities (include if feature involves data)

- Dataset: BloodMNIST image set; attributes include split (train/val/test), image shape, label classes, and dataset version.
- Model: A trained CNN or ViT; attributes include architecture label, configuration, training seed, and checkpoint path.
- Training Run: A specific execution with inputs (config, seed), outputs (artifacts), and timing info; relates to one Model.
- Metrics: Quantitative results per split and class; includes confusion matrix and per-class metrics.
- Visualization Artifact: Overlay images/figures linking back to source samples and corresponding Model.
- Report: Human-readable summary of comparison outcomes with referenced figures and tables.

## Success Criteria (mandatory)

### Measurable Outcomes

- SC-001: End-to-end comparison run (data to results) completes within a documented time budget (target <= 120 minutes for full run; <= 30 minutes in reduced-compute mode).
- SC-002: Both models produce complete metric sets on the designated evaluation split(s), with results reproducible within <= 1.0 pp absolute across two runs with the same seed.
- SC-003: Interpretability overlays are generated for at least 10 samples per class and are human-legible based on a simple visual clarity checklist.
- SC-004: Comparative summary clearly identifies the stronger model per primary metric and states any trade-offs (e.g., accuracy vs. efficiency), verified by a reviewer.
- SC-005: Required artifacts are saved and organized (weights, metrics CSV, figures, report) so a new teammate can locate each within 2 minutes using the provided documentation.
- SC-006: Documentation enables a new user to reproduce results without code changes; a dry run by a reviewer succeeds without assistance.

### Reporting Policy

- Primary results will report both validation and test metrics, highlighting the test split as the primary basis for conclusions. The validation split is used for model selection and sanity checks.

### Time Budget Assumptions

- Full run target: <= 120 minutes on a typical workstation with GPU acceleration; reduced-compute mode target: <= 30 minutes.

