# 🤖 mAIstro Desktop App (Windows)

**mAIstro Desktop** is the easiest way to experience the **mAIstro AI framework** - directly from your desktop.  
It lets you build complete **AI pipelines for medical imaging** without any manual setup or coding.

From **radiomic feature extraction** to **segmentation**, **classification**, **regression**, **evaluation**, and **deployment**, everything happens through an intuitive chat interface where you simply **talk with mAIstro** to create, run, and manage your workflows.

---

## 🚀 Key Features

- 💬 **Chat-based AI development:** Build medical AI pipelines through natural language interaction.  
- 🧩 **End-to-end automation:** Perform **radiomic feature extraction**, **segmentation**, **classification**, **regression**, and **AI model development** in one place.  
- 🧠 **Choose your reasoning engine:** Select between multiple **Large Language Models (LLMs)** as mAIstro’s core engine:
  - GPT-4.1 **(Recommended)** 
  - GPT-4.0  
  - Claude Sonnet 4.5  
  - DeepSeek  
- 🔑 Just import your **API key** for your preferred LLM - no manual configuration needed.  
- ⚙️ **Automatic setup:** The system installs **Miniconda**, creates the `maistro` environment, and installs all dependencies automatically.  
- 🖥️ **GPU-ready** for accelerated workflows.  
- 🪟 Works out of the box on **Windows 10/11**.

---

## 🧰 Prerequisite: Install the Required Compiler

Before running mAIstro for the first time, make sure the Microsoft C++ Build Tools are installed on your system.

1. Go to the official Microsoft page:
    👉 https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Download Build Tools for Visual Studio.

3. During installation, select:

    ✅ “Desktop development with C++”

    Under Optional components, make sure these are checked:

     - MSVC v143 (or newer)

     - Windows 10/11 SDK

---

## 📦 Installation & Launch

1. **Download** the ZIP file:  
   👉 [Download mAIstro Desktop for Windows](<https://drive.google.com/file/d/1c9v7wjvdhNv9L6CFV_OEwELbyY7u3Gx2/view?usp=drive_link>)

   🆕 **Version:** `v1.0.3r` — *released on 8/11/2025*  
   ✨ Includes enhanced capabilities and performance improvements.
    
3. **Unzip** the file anywhere on your computer (recommended: Desktop).  

4. **Double-click** the file:  
   ```bash
   launch_maistro_v1.0.3r.bat
   ```

That’s it! 🎉  
The system will automatically:
- Install **Miniconda** 
- Create the **mAIstro environment**  
- Install all dependencies  
- Launch the **mAIstro Desktop App**

⚠️ **Note:**  
The **first launch** may take up to **15 minutes**, as dependencies are installed.  
Subsequent launches will start instantly.

---

## 💡 Tips

- Avoid spaces in the path.  
- Ensure at least **20 GB** of free disk space.  
- Internet connection is required for the API calls to the LLMs.  
- You can switch between LLMs anytime by updating your API key in the app’s settings.  

---

## 🧩 System Requirements

| Component | Requirement |
|------------|--------------|
| **OS** | Windows 10 or 11 (64-bit) |
| **RAM** | 16 GB (minimum recommended) |
| **GPU** | NVIDIA GPU with CUDA 12.x (optional but recommended) |
| **Storage** | 20 GB free space |
| **Internet** | Required for initial setup and LLM communication |

---

## 🤖 About mAIstro

The **mAIstro framework** is a modular, agentic AI system designed to automate the development of medical imaging workflows.  
It integrates tools for **segmentation**, **feature extraction**, **model training**, and **explainability**, all orchestrated through intelligent **multi-agent reasoning pipelines**.

With **mAIstro Desktop**, you can explore this ecosystem directly on your computer through a single, unified, and intuitive interface.

---

## 📬 Contact

For support or feedback:  
📧 [etzanis@uoc.gr]

---

## Data Sources

The mAIstro Desktop Pack includes example experiments using publicly available open datasets containing CT and MRI scans, as well as clinical and tabular datasets. All datasets are distributed under their respective open licenses and remain the intellectual property of their original authors and institutions. The following datasets were used:

Wasserthal, J., & Akinci D'Antonoli, T. (2025). TotalSegmentator MRI dataset: 616 MRI images with segmentations for 50 anatomical regions (2.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14710732

Wasserthal, J. (2023). Dataset with segmentations of 117 important anatomical structures in 1228 CT images (2.0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10047292

W. Wolberg, O. Mangasarian, N. Street, W. Street. Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository (1993). https://doi.org/10.24432/C5DW2B

Heart Failure Clinical Records [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5Z89R

J.W. Smith, J.E. Everhart, W.C. Dickson, W.C. Knowler, R.S. Johannes. Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proc. Symp Comput. Appl. Med. Care (1988), pp. 261–265. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

R. Detrano et al. International application of a new probability algorithm for the diagnosis of coronary artery disease. Am. J. Cardiol., 64(5), 304–310 (1989). https://archive.ics.uci.edu/dataset/45/heart+disease

A. Kumar. Life Expectancy (WHO) Dataset. Kaggle (2017). https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

J. Yang, R. Shi, D. Wei, et al. MedMNIST v2 – a large-scale lightweight benchmark for 2D and 3D biomedical image classification. Sci. Data, 10 (2023), 41. https://doi.org/10.1038/s41597-022-01721-8

Baid, U., Raza, S.E.A., Saha, A., et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification. arXiv preprint arXiv:2107.02314. https://arxiv.org/abs/2107.02314

Myronenko, A., et al. (2023). Automated 3D Segmentation of Kidneys and Tumors in MICCAI KiTS 2023 Challenge. arXiv preprint arXiv:2310.04110. https://doi.org/10.48550/arXiv.2310.04110

Note:
The datasets above are redistributed solely for educational and research demonstration purposes. Attribution to the original authors and datasets is maintained as required by their respective licenses.
Users of the mAIstro Desktop Pack are responsible for ensuring compliance with each dataset’s license when using, modifying, or redistributing the data.
