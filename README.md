# Heuristic Personality Recognition

Official implementation of the paper:  
**Heuristic personality recognition based on fusing multiple conversations and utterance-level affection**  
*Haijun He, Bobo Li, Yiyun Xiong, Li Zheng, Kang He, Fei Li, Donghong Ji*  
Published in *Information Processing & Management*, 2025.

---

## 🚀 Introduction

This repository provides the source code for our paper.  
We propose a **heuristic personality recognition framework** that:  
- Fuses **multiple conversations** to capture long-term interaction signals.  
- Incorporates **utterance-level affection** to model fine-grained affective cues.  
- Achieves competitive results on benchmark datasets.  

---

## ⚙️ Environment Setup

We recommend using **Python ≥ 3.9** and **PyTorch ≥ 1.12**.  

```bash
git clone https://github.com/<your-repo>/heuristic-personality-recognition.git
cd heuristic-personality-recognition

# Create environment
conda create -n hpr python=3.9
conda activate hpr

# Install dependencies
pip install -r requirements.txt
