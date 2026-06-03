# Echo-eye

Official PyTorch implementation for **Echo-eye**, developed for the study **"From gaze-derived cognitive priors to generative translation of tacit expertise for breast ultrasound AI"**.

Echo-eye is part of a cognition-to-diagnosis framework for breast ultrasound AI. The framework first uses eye-tracking recordings from senior and junior radiologists to model experience-dependent visual search strategies, then converts these strategies into image-conditioned gaze-derived cognitive priors through **AI-Gaze**. Echo-eye subsequently internalizes the generated senior- and junior-derived priors into an image-based diagnostic model through multi-teacher, multi-stage distillation, enabling breast ultrasound diagnosis without gaze input during deployment.

## Overview

This repository contains the core source code for two computational workflows:

1. **AI-Gaze prior generation**: a segmentation-aware gaze generation pipeline that learns from paired breast ultrasound images and radiologist eye-tracking records, then synthesizes senior- and junior-styled gaze-derived cognitive priors for ultrasound images.
2. **Echo-eye diagnostic learning**: a teacher-student training pipeline in which gaze-guided teacher experts transfer complementary senior and junior diagnostic priors to a deployable Student-MoE classifier.

The code supports the main modelling logic described in the manuscript, including gaze-prior generation, teacher expert training, differential knowledge distillation, Student-MoE gating, and model export.

## Study Data

Model development and internal validation used 17,158 breast ultrasound images from Centers A-C and public databases. An independent prospective external cohort of 1,250 images from Centers D and E was reserved for external validation. The study also included radiologist eye-tracking recordings used to characterize senior- and junior-specific visual search behaviour.

The public breast ultrasound datasets used in the study include:

- BrEaST
- BUSBRA
- BUSI
- GDPH&SYSUCC
- QAMEBI
- BUS_UC
- US3M
- BUSC
- UDIAT
- BUS-COT

Private raw ultrasound images, clinical labels, and eye-tracking records are not included in this repository because they contain protected patient information and reader-level behavioural data. Public datasets should be obtained from their original providers.

## Installation

Create a Python environment and install the dependencies:

```bash
conda create -n echo-eye python=3.9
conda activate echo-eye
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build if GPU acceleration is required.

## Repository Structure

```text
Echo-eye/
  scripts/
    generate_public_gaze.py
    train_student_moe.py
  docs/
    code_and_data_availability.md
  requirements.txt
```

## Data Preparation

The scripts expect image classification folders named `benign` and `malignant`. Dataset paths are configured through environment variables, with local defaults under `data/` and generated outputs under `runs/`.

Common environment variables:

```bash
export ECHO_EYE_PUBLIC_BASE=/path/to/public/datasets
export ECHO_EYE_PRIVATE_IMAGE_ROOT=/path/to/private/images
export ECHO_EYE_PRIVATE_GAZE_JUNIOR=/path/to/private/junior_gaze
export ECHO_EYE_PRIVATE_GAZE_SENIOR=/path/to/private/senior_gaze
export ECHO_EYE_SAMUS_CKPT=/path/to/SAMUS.pth
export ECHO_EYE_PUBLIC_GAZE_OUT=/path/to/generated/public_gaze
export ECHO_EYE_CHECKPOINT_DIR=/path/to/checkpoints
```

On Windows PowerShell:

```powershell
$env:ECHO_EYE_PUBLIC_BASE = "D:\path\to\public\datasets"
$env:ECHO_EYE_PRIVATE_IMAGE_ROOT = "D:\path\to\private\images"
$env:ECHO_EYE_PRIVATE_GAZE_JUNIOR = "D:\path\to\private\junior_gaze"
$env:ECHO_EYE_PRIVATE_GAZE_SENIOR = "D:\path\to\private\senior_gaze"
$env:ECHO_EYE_SAMUS_CKPT = "D:\path\to\SAMUS.pth"
$env:ECHO_EYE_PUBLIC_GAZE_OUT = "D:\path\to\generated\public_gaze"
$env:ECHO_EYE_CHECKPOINT_DIR = "D:\path\to\checkpoints"
```

## AI-Gaze Prior Generation

Run:

```bash
python scripts/generate_public_gaze.py
```

This script trains the segmentation and gaze generation workflow, generates pseudo masks when needed, and writes senior- and junior-styled gaze overlays to `ECHO_EYE_PUBLIC_GAZE_OUT`.

## Echo-eye Training

Run:

```bash
python scripts/train_student_moe.py
```

This script trains gaze-guided teacher experts, distills them into image-only student experts, trains the Student-MoE gating network, and exports a deployable `student_moe.pth` package under `ECHO_EYE_CHECKPOINT_DIR`.

## Code and Data Availability

The manuscript-ready code and data availability statement is provided in `docs/code_and_data_availability.md`.

The public source code is intended to be available at:

https://github.com/wyhhxx1121/Echo-eye

Processed private data, trained model weights, and generated gaze-derived cognitive priors may be made available from the corresponding authors upon reasonable request and with permission from the participating hospitals.
