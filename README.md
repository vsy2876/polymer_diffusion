# Polymer SMILES Diffusion Model

This repository contains a generative modeling pipeline for polymer discovery, utilizing a **Conditional Discrete Diffusion Language Model** built on a transformer backbone.

## Overview
Unlike autoregressive models, this implementation treats molecular generation as an iterative denoising process. The model learns to recover polymer SMILES structures from masked sequences, conditioned on physical properties (e.g., band gap, $E_{gc}$).

- **Generative Backbone**: Utilizes `answerdotai/ModernBERT-base` as a transformer encoder-decoder backbone.
- **Conditioning**: Implements property-conditioned generation via a `GaussianFourierProjection` embedding layer, allowing for targeted discovery of polymers with specific band gaps ($E_{gc}$).
- **Training Strategy**: Uses masked diffusion training where the model recovers tokens from varying noise levels, supported by Classifier-Free Guidance (CFG) for improved property adherence.
- **Validation**: Includes a quantum mechanical validation pipeline integrating **PSP (Polymer Structure Predictor)** and **GFN2-xTB** to verify the $E_{gc}$ of generated candidates.

## Getting Started
To replicate the environment and run the pipeline:

1. **Clone the repository:**
```bash
   git clone [https://github.com/vsy2876/polymer_diffusion.git](https://github.com/vsy2876/polymer_diffusion.git)
   cd polymer_diffusion
Core Scripts:training.py: Handles the primary pre-training on the PI1M_v2.csv dataset.finetune_training.py: Fine-tunes the pretrained model on the Egc.csv dataset with property conditioning.finetune_inference.ipynb: Provides a notebook interface for generation, novelty verification, and $E_{gc}$ validation.evaluate_metrics.py: Script for calculating validity, uniqueness, novelty, and property MAE.AttributionThis code is part of a Master's research project at the Georgia Institute of Technology (School of Materials Science & Engineering).© 2026 Vansh Suresh Yadav. All rights reserved.This code is for private research purposes only and may not be copied, modified, or distributed without explicit permission.
