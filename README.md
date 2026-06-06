# Polymer SMILES Diffusion Model

This repository contains a generative modeling pipeline for polymer discovery, utilizing a **Conditional Discrete Diffusion Language Model** built on a transformer backbone.

## Overview
Unlike traditional autoregressive models, this implementation treats molecular generation as an iterative denoising process. The model learns to recover polymer SMILES structures from completely masked sequences, explicitly conditioned on target physical properties.

* **Generative Backbone:** Utilizes `answerdotai/ModernBERT-base` as a bidirectional transformer encoder backbone to capture dense contextual sequence representations.
* **Property Conditioning:** Implements a `GaussianFourierProjection` embedding module that maps scalar property constraints (such as band gap, $E_{gc}$) into high-dimensional frequency spaces to guide the denoising trajectory.
* **Sampling Logic:** Incorporates Classifier-Free Guidance (CFG) during the reverse diffusion process, allowing tuning of how strictly the model adheres to target property thresholds vs. sequence diversity.
* **Validation Matrix:** Integrated evaluation compute calculating standard structural mechanics (Validity, Uniqueness, Novelty) alongside precise quantum-chemical verification using **GFN2-xTB**.

---

## Repository Structure

* `model.py` — Architecture definitions containing the `ConditionalDiffusionLM` model and `GaussianFourierProjection` embedding layers.
* `tokenizer.py` — Vocabulary mappings and regex-based tokenization optimized for handling complex polymer branching and wildcards (`*` / `[*]`).
* `training.py` — Main unconditioned/conditioned pre-training script utilizing the `PI1M_v2.csv` dataset.
* `finetune_training.py` — Fine-tuning pipeline built for strict conditioning on explicit electronic properties via `Egc.csv`.
* `evaluate_metrics.py` — Standardized validation matrix computing internal metrics alongside external quantum-chemical property adherence via the live xTB calculator.
* `finetune_inference.ipynb` — Interactive workspace for checkpoint evaluation, diverse sample stream generation, and property verification.

---

## Getting Started

### 1. Data Requirements
The pipeline expects paths to two primary data files:
* **`PI1M_v2.csv`**: Large-scale polymer database used for capturing structural grammar and baseline synthetic accessibility profiles.
* **`Egc.csv`**: Target electronic properties containing explicit SMILES mapping to calculated band gap values.

### 2. Execution Pipelines
To execute baseline pre-training run:
`python training.py`

To execute property-conditioned fine-tuning for targeted band gap profiles run:
`python finetune_training.py`

---

## Research Attribution
This codebase is a component of ongoing graduate research at the Georgia Institute of Technology (School of Materials Science & Engineering).

**Copyright & Licensing** © 2026 Vansh Suresh Yadav. All rights reserved.  
This code is intended exclusively for private research evaluation. Copying, distributing, or modifying these files without explicit authorization is strictly prohibited.
