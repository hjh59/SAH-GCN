# SAH-GCN
## 🚀 Introduction

This repository provides the source code for our paper.  
We propose a **heuristic personality recognition framework** that:  
- Fuses **multiple conversations** to capture long-term interaction signals.  
- Incorporates **utterance-level affection** to model fine-grained affective cues.  
- Achieves competitive results on benchmark datasets.  

---
## 📂 Dataset
We use the CPED dataset. Please download it from the official repository:

🔗 https://github.com/scutcyr/CPED

After downloading, place the dataset in the following structure:
```arduino
datasets/
  ├── CPED/    # place dataset here
```

## ⚙️ Pretrained Model
Our method relies on pretrained language models (e.g., BERT).
You can download them directly from Hugging Face:

🔗 https://huggingface.co/models

After downloading, place the mdoel in the following structure:
```arduino
models/
  ├── BERT/    # place model here
```

Official implementation of the paper:  
**Heuristic personality recognition based on fusing multiple conversations and utterance-level affection**  
*Haijun He, Bobo Li, Yiyun Xiong, Li Zheng, Kang He, Fei Li, Donghong Ji*  
Published in *Information Processing & Management*, 2025.
```bibtex
@article{he2025heuristic,
  title={Heuristic personality recognition based on fusing multiple conversations and utterance-level affection},
  author={He, Haijun and Li, Bobo and Xiong, Yiyun and Zheng, Li and He, Kang and Li, Fei and Ji, Donghong},
  journal={Information Processing \& Management},
  volume={62},
  number={1},
  pages={103931},
  year={2025},
  publisher={Elsevier}
}

```
---
