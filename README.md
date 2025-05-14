<p align="center">
  <img src="mAIstro_logo.png" alt="mAIstro Logo" width="300"/>
</p>

> An open-source multi-agentic system for automated end-to-end development of radiomics and deep learning models for medical imaging

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

---

## 🚀 About mAIstro
**mAIstro** is an autonomous, open-source multi-agentic system designed to orchestrate the full pipeline of medical imaging AI development - from **exploratory data analysis (EDA)** and **radiomics feature extraction** to **training and deploying deep learning models**.  
Built around a team of specialized agents, mAIstro enables researchers and clinicians to interact with complex AI workflows **using natural language prompts** - no coding required.

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
🔗 Run mAIstro Instantly on Google Colab
Want to try mAIstro without setting up anything locally?
You can now run the full framework directly on Google Colab — no installation required!

👉 [**Launch mAIstro on Colab**](https://colab.research.google.com/drive/1aXNwh9hT9txLIiPetAtIed0_lYiCE3Wv?usp=sharing)

✅ What’s included:
✅ All necessary requirements are automatically installed

✅ The full mAIstro_workspace folder is downloaded, including:

Experiment data

Predefined folders and structure

Ready-to-run example prompts pointing to the correct locations

🔐 What you need to do:
The only manual step is to provide your API key for the LLM of your choice (e.g., OpenAI, Claude, DeepSeek, etc.).

A pre-configured cell is provided with options for multiple LLM providers – just paste your key and you're ready to go!

This makes it easy to explore and test the full functionality of mAIstro on any device, using just your browser.

---

## 📚 Documentation
A full user guide and advanced examples will be provided soon.  
Stay tuned for updates!

---

## 📄 License

This project is licensed under the **Apache License 2.0**.  
You are free to use, modify, and distribute this software under the terms of the license.

---

### 📚 Cite this work

If you use **mAIstro** in your research, please cite:

>Tzanis E., Klontzas M. E. (2025). *mAIstro: an open-source multi-agentic system for automated end-to-end development of radiomics and deep learning models for medical imaging*. arXiv: [2505.03785](https://arxiv.org/abs/2505.03785), DOI: [https://doi.org/10.48550/arXiv.2505.03785](https://doi.org/10.48550/arXiv.2505.03785)

---

