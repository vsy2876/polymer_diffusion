# Conditional Discrete Diffusion Language Model for Polymer Discovery

This repository contains the core generative modeling pipeline for property-conditioned polymer discovery, leveraging a Conditional Discrete Diffusion Language Model (CDDLM) built on a modern transformer encoder backbone.

## Overview
Unlike traditional autoregressive sequence models, this implementation frames molecular generation as a non-autoregressive, iterative denoising process. The model learns to recover clean polymer SMILES structures from completely corrupted (masked) sequences, explicitly guided by continuous physical property constraints.

* Generative Backbone: Utilizes answerdotai/ModernBERT-base as a bidirectional transformer encoder to capture dense, long-range contextual sequence representations across complex macromolecular architectures.
* Property Conditioning: Implements a continuous GaussianFourierProjection embedding module that maps scalar property constraints—such as target electronic band gap (E_g)—into a high-dimensional frequency space to drive the reverse diffusion trajectory.
* Sampling Strategy: Incorporates Classifier-Free Guidance (CFG) during the reverse denoising process, enabling precise control over the trade-off between target property adherence and sequence diversity.
* Validation Pipeline: Features an integrated evaluation suite calculating standard structural metrics (Validity, Uniqueness, Novelty) alongside real-time quantum-chemical verification via a live GFN2-xTB calculator.

---

## Repository Structure

├── .gitignore
├── README.md
├── model.py                 # Core CDDLM architecture and GaussianFourierProjection layers
├── tokenizer.py             # Regex-based tokenizer optimized for polymer branching/wildcards (* / [*])
├── training.py              # Unconditioned/conditioned baseline pre-training loop
├── finetune_training.py     # Property-conditioned fine-tuning pipeline for explicit target profiles
├── evaluate_metrics.py      # Validation matrix computing structural metrics and live xTB property tracking
├── finetune_inference.ipynb # Interactive evaluation, sample generation streams, and property checking
├── train.sh                 # Pre-training execution script
└── finetune.sh              # Fine-tuning execution script

---

## Getting Started

### 1. Data Requirements
The pipeline expects paths to two primary data tracking files:
* `PI1M_v2.csv`: Large-scale polymer database used for capturing structural syntax and baseline synthetic accessibility profiles.
* `Egc.csv`: Target dataset containing explicit SMILES mappings to calculated electronic band gap (E_g) values.

### 2. Execution
To run baseline pre-training:
python training.py (or run: bash train.sh)

To execute property-conditioned fine-tuning for targeted electronic profiles:
python finetune_training.py (or run: bash finetune.sh)

---

## Research Attribution
This codebase is a component of ongoing graduate research at the Georgia Institute of Technology (School of Materials Science & Engineering).

Copyright & Licensing
© 2026 Vansh Suresh Yadav. All rights reserved.
This code is intended exclusively for private research evaluation. Copying, distributing, or modifying these files without explicit authorization is strictly prohibited.

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
