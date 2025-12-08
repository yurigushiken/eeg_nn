# EEG Deep Learning for Numerical Cognition

A deep learning pipeline for decoding numerical representations from 128-channel EEG data. This project implements **Deep Learning-based Representational Similarity Analysis (RSA)** to map the neural geometry of the Parallel Individuation (PI) and Approximate Number System (ANS).

## Scientific Motivation

### The Two-Systems Hypothesis

Cognitive neuroscience proposes two distinct systems for processing quantities:

1. **Parallel Individuation (PI):** Processes small numbers via rapid, precise "object files."
2. **Approximate Number System (ANS):** Processes larger numbers via magnitude estimation, where precision follows Weber's law.

### This project asks

Can we use Convolutional Neural Networks (CNNs) to map the representational state space of numerosity?

And:
- **The Boundary:** Can we reveal patterns that suggest a divide between 'small' and 'large' numbers?
- **The Structure:** Is there a unique representational geometry within the small-number range?
- **Grouping:** Does the brain use grouping mechanisms to represent composite numbers?

## Study Background

This project analyzes data from a numerical oddball task (N=24 adults, 6,480 trials). Participants viewed dot arrays while EEG was recorded. We apply **Deep Learning RSA** to decode fine-grained spatiotemporal patterns from raw, single-trial data.

## What This Pipeline Does

This repository implements a pipeline that uses a compact CNN (**EEGNeX**) as a distance metric for Representational Similarity Analysis.

Representational Similarity Analysis (RSA):
1. Train neural networks to distinguish every possible pair of numerosities.
2. Use **decoding accuracy** as the measure of representational dissimilarity.
3. Construct a **Representational Dissimilarity Matrix (RDM)** to visualize the neural geometry.
4. Control for visual confounds (pixel area) using partial correlation analysis.

## Findings

### Uncovering the Neural State Space

By projecting our Deep Learning RDM into 2D space (Multidimensional Scaling), we uncovered a non-linear architecture of number processing in the adult brain.

**1. System Distinctness & Boundary:**  
...

**2. Structure Within Object Tracking:**  
The subitizing range is not composed of uniformly distinct slots. Numerosities 2-4 form a similarity cluster.

**3. The Divisibility Effect:**  
High confusion between numerosities 5 and 6. 

### Robustness to Visual Confounds
We performed a control analysis to ensure these results were not driven by low-level visual features.

## Supported Workflows

The pipeline supports end-to-end RSA execution:

* 
sa_binary: Trains pairwise classifiers across 10 random seeds using Leave-One-Subject-Out (LOSO) cross-validation.
* 
sa_pixel_control: Performs post-hoc statistical analysis to regress out pixel confounds from the brain RDM.
* generate_rsa_tables: Produces publication-ready LaTeX tables of pairwise decoding statistics (Holm-corrected).

## Features

- **Leak-Free Validation:** Subject-aware splits ensure no participant data appears in both train and test.
- **Constitutional Rigor:** All parameters must be explicitly specified via YAML.
- **Automated RSA:** End-to-end scripts for training, RDM generation, and Multidimensional Scaling (MDS) visualization.
- **Explainable AI:** Integrated Gradients highlight spatiotemporal feature importance.
- **Statistical Rigor:** Deterministic seeding, permutation testing, and partial correlation analysis for confounds.
- **Full Provenance:** Every run logs model class, library versions, hardware, and seeds.

  </div>
</details>
