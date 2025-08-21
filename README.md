# Bangla Compound Character Classification  
*Using CNN with Multi-Level Self-Attention on the MatriVasha Dataset*

This is my final project for the Neural Networks course. The goal was to classify 120 different **Bangla handwritten compound characters** using a custom-built deep learning pipeline that combines a CNN backbone with attention mechanisms. I also included heatmap visualizations to show where the model was focusing during classification.

---

## Overview

This project is split into two main stages:

1. **Pretraining**: A convolutional neural network (CNN) is trained on CIFAR-100 (grayscale) to learn general image features.
2. **Finetuning**: The pretrained model is extended with **multi-level self-attention modules** and fine-tuned on the MatriVasha dataset of Bangla handwritten compound characters.

Due to **resource limitations** (running on Google Colab), training was done for only a few epochs — but even then, the model produced solid results and insightful attention maps.

---

## Dataset

This project uses the **MatriVasha** dataset, a large-scale handwritten Bangla compound character dataset containing 120 classes collected from male and female contributors.

- 🔗 [Download the dataset on Mendeley Data](https://doi.org/10.17632/v39pc2g2wp.1)

**Citation**:

Ferdous, Jannatul; Karmaker, Suvrajit; Rabby, AKM Shahariar Azad; Hossain, Syed Akhter  
(2021), *“MatriVasha: Bangla Handwritten Compound Character Dataset and Recognition”*,  
Mendeley Data, V1, https://doi.org/10.17632/v39pc2g2wp.1

> Dataset was manually preprocessed by merging male/female folders, resizing to 128×128, inverting pixel intensities, and normalizing to [-1, 1].

---

## Model Architecture

### 📌 Stage 1: `CustomCNN` (Pretraining)
- 4-layer convolutional network
- Trained on grayscale CIFAR-100
- Used `AdamW` optimizer and Xavier initialization

### 📌 Stage 2: `CNNEncoderWithAttention` (Finetuning)
- Self-attention modules applied on three CNN layers:
  - 32×128×128 features
  - 64×64×64 features
  - 128×32×32 features
- Each attention block projects features and computes a learned focus map
- Outputs from all three are concatenated and passed through a classifier (120 outputs)

---

## Training Details

| Phase        | Dataset     | Epochs | Batch Size | Notes                          |
|--------------|-------------|--------|------------|--------------------------------|
| Pretraining  | CIFAR-100   | 5      | 512        | Grayscale images               |
| Finetuning   | MatriVasha  | 2      | 64         | Due to Colab limitations 😅    |

Loss curves are plotted for both training and validation steps.

---

## Attention Visualization

The project includes visualization tools to inspect the model’s focus:

- Heatmaps from attention maps overlaid on the input image
- You can view:
  - Correct predictions
  - Incorrect predictions
  - One correct & incorrect prediction **per class**
- Confidence scores printed for each prediction

This gives a window into what the model “sees” and whether it’s making decisions for the right reasons.

---

## Evaluation Metrics

- **Confusion matrix** using Seaborn
- **Classification report** with precision, recall, and F1-score
- Outputs and errors shown with heatmap overlays

Despite the short training window, the model learned meaningful patterns across all 120 classes.

---

## Presentation

You can view the slides summarizing this project here:  
[🖼️ Bangla Compound Character Classification – Canva Presentation](https://www.canva.com/design/DAGbsHpjimM/YBabg8wrRUIBukZDs-MGCg/view?utm_content=DAGbsHpjimM&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hd93bac0ca5)

---

## Run on Google Colab

You can open and run the full notebook directly on Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hNxYmXj8ve6q1TrD90tX4R55YoPnY5Ju?usp=sharing)

---

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
