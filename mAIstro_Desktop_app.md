# 🤖 mAIstro Desktop App (Windows)

**mAIstro Desktop** is the easiest way to experience the **mAIstro AI framework** - directly from your desktop.  
It lets you build complete **AI pipelines for medical imaging** without any manual setup or coding.

From **radiomic feature extraction** to **segmentation**, **classification**, **regression**, **evaluation**, and **deployment**, everything happens through an intuitive chat interface where you simply **talk with mAIstro** to create, run, and manage your workflows.

---

## 🚀 Key Features

- 💬 **Chat-based AI development:** Build medical AI pipelines through natural language interaction.  
- 🧩 **End-to-end automation:** Perform **radiomic feature extraction**, **segmentation**, **classification**, **regression**, and **AI model development** in one place.  
- 🧠 **Choose your reasoning engine:** Select between multiple **Large Language Models (LLMs)** as mAIstro’s core engine:
  - GPT-4.1  
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
   👉 [Download mAIstro Desktop for Windows](<https://drive.google.com/file/d/1wd4is1-EA21V5SFyz9RuGhs4r31w9hQI/view?usp=sharing>)

2. **Unzip** the file anywhere on your computer (recommended: Desktop).  

3. **Double-click** the file:  
   ```bash
   launch_maistro_v1.0r.bat
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
- Internet connection is required during the first setup.  
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
