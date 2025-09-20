
# [SAH-GCN](https://github.com/hjh59/SAH-GCN)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ipm.2024.103931-blue.svg)](https://doi.org/10.1016/j.ipm.2024.103931)


## 🚀 Introduction
This repository provides the source code for our paper.  
We propose a **heuristic personality recognition framework** that:  
- Fuses multiple conversations to capture more abundant interaction signals.  
- Incorporates utterance-level affection to model fine-grained affective cues.  
- Achieves competitive results on benchmark datasets.  

---
## 📂 Dataset
We use the CPED dataset. Please download it from the official repository:  

🔗 https://github.com/scutcyr/CPED

After downloading, place the dataset in the following structure:
```bash
datasets/
  ├── CPED/    # place dataset here
```

## ⚙️ Pretrained Model
Our method relies on pretrained language models (e.g., BERT).
You can download them directly from Hugging Face:

🔗 https://huggingface.co/models

After downloading, place the model in the following structure:
```bash
pretrained_model/
  ├── BERT/    # place model here
```

## Corresponding paper:  
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
  publisher={Elsevier},
  doi={10.1016/j.ipm.2024.103931}
}
```
---
