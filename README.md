# EEG Deep Learning for Numerical Cognition

A deep learning pipeline for decoding numerical representations from 128-channel EEG data. This project implements **Deep Learning-based Representational Similarity Analysis (RSA)** to map the neural geometry of the Parallel Individuation (PI) and Approximate Number System (ANS).

## Scientific Motivation

### The Two-Systems Hypothesis

Cognitive neuroscience proposes two distinct systems for processing quantities:

1. **Parallel Individuation (PI):** Processes small numbers via rapid, precise "object files."
2. **Approximate Number System (ANS):** Processes larger numbers via magnitude estimation, where precision follows Weber's law.

### The Geometric Debate

While the existence of these systems is established, the precise neural geometry of the transition remains debated. Does the brain switch systems abruptly? Are small numbers represented as distinct "slots," or do they form a categorical cluster?

**This project asks:** Can we use Convolutional Neural Networks (CNNs) to map the representational state space of numerosity?

Specifically:
- **The Boundary:** Can we decode the exact transition between the precision of PI and the approximation of ANS?
- **The Structure:** Is there a unique representational geometry within the small-number range?
- **Grouping:** Does the brain utilize grouping mechanisms to represent composite numbers?

## Study Background

This project analyzes data from a numerical oddball task (N=24 adults, 6,480 trials). Participants viewed dot arrays while EEG was recorded. Unlike traditional ERP analyses that average signals across electrodes, we apply **Deep Learning RSA** to decode fine-grained spatiotemporal patterns from raw, single-trial data.

## What This Pipeline Does

This repository implements a rigorous pipeline that uses a compact CNN (**EEGNeX**) as a distance metric for Representational Similarity Analysis.

Instead of standard correlation-based RSA, we:
1.  Train neural networks to distinguish every possible pair of numerosities.
2.  Use **decoding accuracy** as the measure of representational dissimilarity.
3.  Construct a **Representational Dissimilarity Matrix (RDM)** to visualize the neural geometry.
4.  Control for visual confounds (pixel area) using partial correlation analysis.

## Findings

### Uncovering the Neural State Space

By projecting our Deep Learning RDM into 2D space (Multidimensional Scaling), we uncovered a non-linear architecture of number processing in the adult brain.

**1. System Distinctness & Boundary:**
We identified a representational boundary that delineates the limit of the object-tracking system. Confirm the existence of two distinct neural codes for small versus large quantities, with a clear transition between them.

**2. Structure Within Object Tracking:**
we do not find uniform distinctness for all small numbers. We find a similarity cluster.

**3. The Divisibility Effect:**
(awaiting results)

### Robustness to Visual Confounds
We performed a rigorous control analysis to ensure these results were not driven by low-level visual features. 

## Supported Workflows

The pipeline supports end-to-end RSA execution:

* `rsa_binary`: Trains pairwise classifiers across 10 random seeds using Leave-One-Subject-Out (LOSO) cross-validation.
* `rsa_pixel_control`: Performs post-hoc statistical analysis to regress out pixel confounds from the brain RDM.
* `generate_rsa_tables`: Produces publication-ready LaTeX tables of pairwise decoding statistics (Holm-corrected).

## Features

- **Leak-Free Validation:** Subject-aware splits ensure no participant data appears in both train and test.
- **Constitutional Rigor:** All parameters must be explicitly specified via YAML.
- **Automated RSA:** End-to-end scripts for training, RDM generation, and Multidimensional Scaling (MDS) visualization.
- **Explainable AI:** Integrated Gradients to map spatiotemporal feature importance.
- **Statistical Rigor:** Deterministic seeding, permutation testing, and partial correlation analysis for confounds.
- **Full Provenance:** Every run logs model class, library versions, hardware, and seeds.

