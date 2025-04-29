
# 🎼 mAIstro
> An open-source multi-agentic system for automated end-to-end development of radiomics and deep learning models for medical imaging

![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)  

---

## 🚀 About mAIstro
**mAIstro** is a fully autonomous, open-source multi-agentic system designed to orchestrate the full pipeline of medical imaging AI development — from **exploratory data analysis (EDA)** and **radiomics feature extraction** to **training and deploying deep learning models**.  
Built around a team of specialized agents, mAIstro enables researchers and clinicians to interact with complex AI workflows **using natural language prompts** — no coding required.

---

## 🧩 Key Features
- 🔎 Autonomous **Exploratory Data Analysis (EDA)**
- 🧠 **Radiomics feature extraction** (for CT, MRI, and multi-parametric imaging)
- 🏥 **nnU-Net Agent** for medical image segmentation
- 🏥 **TotalSegmentator Agent** for full-body and organ-specific segmentation
- 🖼️ **Image Classification Agent** (ResNet, VGG16, InceptionV3 architectures)
- 📊 **Feature Importance and Feature Selection**
- 📈 **Tabular data Classification and Regression Agents**
- 🔗 Modular tool-based architecture for extensibility
- 📝 Integrated in a single user-friendly Jupyter Notebook

---

## ⚙️ Installation and Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/mAIstro.git
cd mAIstro
```

2. **(Recommended) Create and activate a virtual environment**:
```bash
python -m venv maistro-env
source maistro-env/bin/activate          # On Linux/macOS
.\maistro-env\Scriptsctivate            # On Windows
```

3. **Install required Python packages**:
```bash
pip install -r requirements.txt
```

4. **Set up nnU-Net environment variables**:  
   To enable the nnU-Net Agent, you must configure the paths following the [nnU-Net setup instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

Example setup:
```bash
export nnUNet_raw_data_base="/path/to/nnUNet_raw_data_base"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export RESULTS_FOLDER="/path/to/nnUNet_trained_models"
```
(For Windows users, use `set` instead of `export`.)

---

## 🧠 How to Use

- Open and run the `mAIstro.ipynb` notebook.
- Choose a natural language prompt corresponding to your task (e.g., train a model, extract radiomics, perform EDA).
- mAIstro's agents will autonomously reason, select the right tools, execute the task, and generate outputs.
- All results (plots, trained models, reports) will be saved automatically in specified directories.

---

## 📚 Documentation
A full user guide and advanced examples will be provided soon.  
Stay tuned for updates!

---

## 🤝 Contributing

Contributions are welcome!  
Please feel free to fork the repository and submit pull requests. For major changes, open an issue first to discuss your ideas.

---

## 📄 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 🧡 Acknowledgments

- 🤖 Huggingface [`smolagents`](https://github.com/huggingface/smolagents) for lightweight agentic abstractions
- 🏥 [`nnU-Net`](https://github.com/MIC-DKFZ/nnUNet) for segmentation pipelines
- 🏥 [`TotalSegmentator`](https://github.com/wasserth/TotalSegmentator) for multi-organ segmentation
- 🧬 [`PyRadiomics`](https://github.com/Radiomics/pyradiomics) for radiomics feature extraction
- 📊 [`PyCaret`](https://github.com/pycaret/pycaret) for tabular data modeling

---
