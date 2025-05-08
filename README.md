<p align="center">
  <img src="mAIstro_logo.png" alt="mAIstro Logo" width="300"/>
</p>

> An open-source multi-agentic system for automated end-to-end development of radiomics and deep learning models for medical imaging

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

---

## 🚀 About mAIstro
**mAIstro** is an autonomous, open-source multi-agentic system designed to orchestrate the full pipeline of medical imaging AI development — from **exploratory data analysis (EDA)** and **radiomics feature extraction** to **training and deploying deep learning models**.  
Built around a team of specialized agents, mAIstro enables researchers and clinicians to interact with complex AI workflows **using natural language prompts** — no coding required.

🌐 **LLM-Agnostic Design**: mAIstro can operate with both open-source and commercial LLMs (e.g., GPT-4, Claude, DeepSeek, LLaMA, Qwen), providing flexibility across environments.

Built on top of 🤗 Hugging Face's [smolagents](https://github.com/huggingface/smolagents) framework.

---

## ✨  Key Features
- 🔎 Autonomous **Exploratory Data Analysis (EDA)**
- 🧬 **Radiomics feature extraction** (for CT, MRI, and multi-parametric imaging)
- ⚙️ **nnU-Net Agent** for segmentation model development and implementation
- ⚙️ **TotalSegmentator Agent** for full-body and organ-specific automatic segmentation
- 🩻 **Image Classification Agent** (ResNet, VGG16, InceptionV3 architectures)
- 📊 **Feature Importance and Feature Selection**
- 📈 **Tabular data Classification and Regression Agents**
- 🛠️ Modular tool-based architecture for extensibility
- 🧾 Integrated in a single user-friendly Jupyter Notebook

---

## ⚙️ Instructions to Set Up Docker and Run the mAIstro Environment

👉 [Instructions to set up Docker and run mAIstro](./install_mAIstro_env.md)

---

## 📚 Documentation
A full user guide and advanced examples will be provided soon.  
Stay tuned for updates!

---

## 📄 License

This project is licensed under the **Apache License 2.0**.  
You are free to use, modify, and distribute this software under the terms of the license.

---

## 🧡 Acknowledgments

- 🤗 Huggingface [`smolagents`](https://github.com/huggingface/smolagents) for lightweight agentic abstractions
- ⚙️ [`nnU-Net`](https://github.com/MIC-DKFZ/nnUNet) for segmentation pipelines
- ⚙️ [`TotalSegmentator`](https://github.com/wasserth/TotalSegmentator) for multi-organ segmentation
- 🧬 [`PyRadiomics`](https://github.com/Radiomics/pyradiomics) for radiomics feature extraction
- 📊 [`PyCaret`](https://github.com/pycaret/pycaret) for tabular data modeling

---

### 📚 Cite this work

If you use **mAIstro** in your research, please cite:

>Tzanis E., Klontzas M. E. (2025). *mAIstro: an open-source multi-agentic system for automated end-to-end development of radiomics and deep learning models for medical imaging*. arXiv: [2505.03785](https://arxiv.org/abs/2505.03785), DOI: [https://doi.org/10.48550/arXiv.2505.03785](https://doi.org/10.48550/arXiv.2505.03785)

---

